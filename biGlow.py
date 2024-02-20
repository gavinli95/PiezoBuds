import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))
'''
biGlow takes the input of 2 different modalities, and use one of it to train the other.
'''

class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / (self.scale - self.loc + 1e-8)


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(np.copy(w_s))
        w_u = torch.from_numpy(np.copy(w_u))

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True, condition_size=None, use_bi_flow=False):
        super().__init__()

        self.affine = affine
        self.condition_size = condition_size
        self.use_bi_flow = use_bi_flow

        if condition_size == None:
            self.net = nn.Sequential(
                nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(filter_size, filter_size, 1),
                nn.ReLU(inplace=True),
                ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channel // 2 + condition_size, filter_size, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(filter_size, filter_size, 1),
                nn.ReLU(inplace=True),
                ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
            )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        if self.condition_size != None:
            input, condition = input
        
        # split the input into 2 parts
        in_a, in_b = input.chunk(2, 1)

        if self.condition_size != None:
            # Condition should be a 1D vector per batch, we expand it to match the spatial dimensions of in_a
            # condition = condition.view(condition.size(0), self.condition_size, 1, 1)
            # condition = condition.expand(-1, -1, in_a.size(2), in_a.size(3))
            if self.use_bi_flow:
                b, c, h, w = condition.shape
                condition_reshape = condition.view(b, c, h // 2, 2, w // 2, 2)
                condition_reshape = condition_reshape.permute(0, 1, 3, 5, 2, 4)
                condition_final = condition_reshape.contiguous().view(b, self.condition_size, in_a.size(2), in_a.size(3))
            else:
                condition_final = condition
            in_a_conditioned = torch.cat((in_a, condition_final), 1)

        if self.affine:
            if self.condition_size != None:
                log_s, t = self.net(in_a_conditioned).chunk(2, 1)
            else:
                log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            if self.condition_size != None:
                net_out = self.net(in_a_conditioned)
            else:
                net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        if self.condition_size != None:
            output, condition = output
        out_a, out_b = output.chunk(2, 1)
        
        if self.condition_size != None:
            # Condition should be a 1D vector per batch, we expand it to match the spatial dimensions of in_a
            # condition = condition.view(condition.size(0), self.condition_size, 1, 1)
            # condition = condition.expand(-1, -1, out_a.size(2), out_a.size(3))
            if self.use_bi_flow:
                b, c, h, w = condition.shape
                condition_reshape = condition.view(b, c, h // 2, 2, w // 2, 2)
                condition_reshape = condition_reshape.permute(0, 1, 3, 5, 2, 4)
                condition_final = condition_reshape.contiguous().view(b, self.condition_size, out_a.size(2), out_a.size(3))
            else:
                condition_final = condition
            out_a_conditioned = torch.cat((out_a, condition_final), 1)

        if self.affine:
            if self.condition_size != None:
                log_s, t = self.net(out_a_conditioned).chunk(2, 1)
            else:
                log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / (s - t + 1e-8)

        else:
            if self.condition_size != None:
                net_out = self.net(out_a_conditioned)
            else:
                net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True, condition_size=None, use_bi_flow=True):
        super().__init__()

        self.condition_size = condition_size
        self.use_bi_flow = use_bi_flow

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine, condition_size=condition_size, use_bi_flow=use_bi_flow)

    def forward(self, input):
        # if we use condition here, input should be a tuple of original input and condition
        # otherwise, it should only contain input itself
        if self.condition_size != None:
            input, condition = input
        
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        if self.condition_size != None:
            out, det2 = self.coupling((out, condition))
        else:
            out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        if self.condition_size != None:
            output, condition = output
        if self.condition_size != None:
            input = self.coupling.reverse((output, condition))
        else:
            input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / (torch.exp(2 * log_sd) + 1e-5)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True, condition_size=None, use_bi_flow=True):
        super().__init__()

        squeeze_dim = in_channel * 4
        self.condition_size = condition_size
        self.use_bi_flow = use_bi_flow

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu, condition_size=condition_size, use_bi_flow=use_bi_flow))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        if self.condition_size != None:
            input, condition = input
        
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            if self.condition_size != None:
                out, det = flow((out, condition))
            else:
                out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        if self.condition_size != None:
            output, condition = output
        
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            if self.condition_size != None:
                input = flow.reverse((input, condition))
            else:
                input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed

class conditionGlow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True, use_bi_flow=False
    ):
        super().__init__()
        self.blocks_input = nn.ModuleList()
        self.blocks_condition = nn.ModuleList()
        self.n_block = n_block

        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks_condition.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, condition_size=None, use_bi_flow=use_bi_flow))
            self.blocks_input.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, condition_size=n_channel * 2, use_bi_flow=use_bi_flow))
            n_channel *= 2
 
        self.blocks_condition.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, split=False, condition_size=None, use_bi_flow=use_bi_flow))
        self.blocks_input.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, split=False, condition_size=n_channel * 4, use_bi_flow=use_bi_flow))

    def forward(self, input, condition):
    # def forward(self, inputs):
        # n, _, _, _ = inputs.shape
        # input, condition = torch.split(inputs, n // 2, dim=0)
        log_p_sum_i, log_p_sum_c = 0, 0
        logdet_i, logdet_c = 0, 0
        z_outs_i, z_outs_c = [], []

        out_i = input
        out_c = condition
        for i in range(self.n_block):
            block_con = self.blocks_condition[i]
            block_ipt = self.blocks_input[i]

            out_c, det_c, log_p_c, z_new_c = block_con(out_c)
            out_i, det_i, log_p_i, z_new_i = block_ipt((out_i, z_new_c))
            
            z_outs_i.append(z_new_i)
            z_outs_c.append(z_new_c)

            logdet_i += det_i
            logdet_c += det_c

            if log_p_i is not None:
                log_p_sum_i +=log_p_i
            if log_p_c is not None:
                log_p_sum_c +=log_p_c

        return ((log_p_sum_i, log_p_sum_c), (logdet_i, logdet_c), (z_outs_i, z_outs_c))

    def reverse(self, z_list, reconstruct=False):
        z_list, condition_list = z_list
        for i, block in enumerate(self.blocks_input[::-1]):
            if i == 0:
                input = block.reverse((z_list[-1], condition_list[-1]), z_list[-1], reconstruct=reconstruct)
            else:
                input = block.reverse((input, condition_list[-(i + 1)]), z_list[-(i + 1)], reconstruct=reconstruct)
        
        return input
    
class biGlow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True, use_bi_flow=True
    ):
        super().__init__()
        self.blocks_input = nn.ModuleList()
        self.blocks_condition = nn.ModuleList()
        self.n_block = n_block

        n_channel = in_channel
        for i in range(n_block - 1):
            if i == 0:
                self.blocks_condition.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, condition_size=None, use_bi_flow=use_bi_flow))
                self.blocks_input.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, condition_size=None, use_bi_flow=use_bi_flow))
            else:
                self.blocks_condition.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, condition_size=n_channel * 4, use_bi_flow=use_bi_flow))
                self.blocks_input.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, condition_size=n_channel * 4, use_bi_flow=use_bi_flow))
            n_channel *= 2
 
        self.blocks_condition.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, split=False, condition_size=n_channel * 4, use_bi_flow=use_bi_flow))
        self.blocks_input.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, split=False, condition_size=n_channel * 4, use_bi_flow=use_bi_flow))

    def forward(self, input, condition):
        log_p_sum_i, log_p_sum_c = 0, 0
        logdet_i, logdet_c = 0, 0
        z_outs_i, z_outs_c = [], []

        out_i = input
        out_c = condition
        for i in range(self.n_block):
            block_con = self.blocks_condition[i]
            block_ipt = self.blocks_input[i]

            if i == 0:
                out_c, det_c, log_p_c, z_new_c_new = block_con(out_c)
                out_i, det_i, log_p_i, z_new_i_new = block_ipt(out_i)
            else:
                out_c, det_c, log_p_c, z_new_c_new = block_con((out_c, z_new_i_old))
                out_i, det_i, log_p_i, z_new_i_new = block_ipt((out_i, z_new_c_old))
            
            z_outs_i.append(z_new_i_new)
            z_outs_c.append(z_new_c_new)
            
            z_new_i_old = z_new_i_new
            z_new_c_old = z_new_c_new

            logdet_i += det_i
            logdet_c += det_c

            if log_p_i is not None:
                log_p_sum_i +=log_p_i
            if log_p_c is not None:
                log_p_sum_c +=log_p_c

        return (log_p_sum_i, log_p_sum_c), (logdet_i, logdet_c), (z_outs_i, z_outs_c)

    def reverse(self, z_list, reconstruct=False):
        z_list, condition_list = z_list
        for i, block_i in enumerate(self.blocks_input[::-1]):
            block_c = self.blocks_condition[-(i + 1)]
            if i == len(condition_list) - 1:
                input_i = block_i.reverse(z_list[-(i + 1)], z_list[-(i + 1)], reconstruct=reconstruct)
                input_c = block_c.reverse(condition_list[-(i + 1)], condition_list[-(i + 1)], reconstruct=reconstruct)
            else:
                input_i = block_i.reverse((z_list[-(i + 1)], condition_list[-(i + 2)]), z_list[-(i + 1)], reconstruct=reconstruct)
                input_c = block_c.reverse((condition_list[-(i + 1)], z_list[-(i + 2)]), condition_list[-(i + 1)], reconstruct=reconstruct)
        
        return (input_i, input_c)

if __name__=='__main__':
    input = torch.rand((10, 3, 8, 8))
    target = torch.rand((10, 3, 8, 8))
    model = conditionGlow(in_channel=3, n_flow=4, n_block=3)
    log_p_sum, logdet, z_outs = model(input, target)
    recovers = model.reverse(z_outs, reconstruct=False)
    print(model)
    print(recovers.shape)