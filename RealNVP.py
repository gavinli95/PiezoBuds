import torch
from torch import nn

# First, we need to adjust the CouplingLayer to accept the conditional input X.
class ConditionalCouplingLayer(nn.Module):
    def __init__(self, input_dim, condition_dim, device):
        super(ConditionalCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        # The network now accepts concatenated input of X and Y to predict the parameters
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim * 2)  # Outputs both scale and translation parameters for each dimension
        )
        
        # Initialize masks
        self.mask = torch.arange(self.input_dim) % 2
        self.mask = self.mask.float().unsqueeze(0)
        self.mask = self.mask.to(device)

    def forward(self, x, condition):
        # Concatenate the condition and the input
        combined = torch.cat([x * self.mask, condition], dim=1)
        log_s, t = self.net(combined).chunk(2, dim=1)
        s = torch.exp(log_s) * (1 - self.mask)
        t = t * (1 - self.mask)
        y = x * (1 - self.mask) + (x * s + t) * self.mask
        log_det_jacobian = log_s.sum(dim=1)
        return y, log_det_jacobian

    def inverse(self, y, condition):
        # Concatenate the condition and the input
        combined = torch.cat([y * self.mask, condition], dim=1)
        log_s, t = self.net(combined).chunk(2, dim=1)
        s = torch.exp(-log_s) * (1 - self.mask)
        t = t * (1 - self.mask)
        x = y * (1 - self.mask) + (y - t) * s * self.mask
        return x

# Next, we adjust the RealNVP model to include the conditional input.
class ConditionalRealNVP(nn.Module):
    def __init__(self, input_dim, condition_dim, num_coupling_layers, device):
        super(ConditionalRealNVP, self).__init__()
        self.coupling_layers = nn.ModuleList([
            ConditionalCouplingLayer(input_dim, condition_dim, device)
            for _ in range(num_coupling_layers)
        ])

    def forward(self, x, condition):
        log_det_jacobian = 0
        for layer in self.coupling_layers:
            x, log_det_j = layer(x, condition)
            log_det_jacobian += log_det_j
        return x, log_det_jacobian

    def inverse(self, z, condition):
        for layer in reversed(self.coupling_layers):
            z = layer.inverse(z, condition)
        return z

# Define the conditional NLL loss function.
def conditional_nll_loss(x, y, model):
    '''
    x: condition
    y: input
    '''
    z, log_det_jacobian = model(y, x)
    log_base_distribution = -0.5 * z.pow(2).sum(1)  # Assuming a standard normal base distribution
    log_base_distribution = log_base_distribution.to(next(model.parameters()).device)
    nll = -log_base_distribution - log_det_jacobian
    return nll.mean()


if __name__ == '__main__':
    # Let's create the model instance with the new conditional setup.
    input_dim = 192  # This is the D in the shape (B, D)
    condition_dim = 192  # Assuming the condition is also the same dimension as the input
    num_coupling_layers = 3

    # Instantiate the conditional RealNVP model.
    conditional_model = ConditionalRealNVP(input_dim, condition_dim, num_coupling_layers)

    # Define the optimizer.
    optimizer = torch.optim.Adam(conditional_model.parameters(), lr=0.001)

    # Simulate a single training step with the conditional model.

    # Zero the gradients before running the backward pass.
    X = torch.rand((10, 192))
    Y = torch.rand((10, 192))

    z, _ = conditional_model(Y, X)
    print(z.shape)
    
    optimizer.zero_grad()

    # Compute the conditional loss.
    loss = conditional_nll_loss(X, Y, conditional_model)

    # Backpropagation to compute gradients.
    loss.backward()

    # Update the weights.
    optimizer.step()

    # Return the loss value for inspection.
    loss.item()
    