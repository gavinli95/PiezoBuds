'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax, AAMsoftmax576
from model import ECAPA_TDNN
import random
from biGlow import *
from utils import *
from dataLoader import *

class PiezoBudsModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, device, num_frames, **kwargs):
		super(PiezoBudsModel, self).__init__()
		self.device = device
		self.user_list = [i for i in range(n_class)]
		self.num_frames = num_frames

		## Extractor
		self.encoder_a = ECAPA_TDNN(C = C).to(device)
		self.encoder_p = ECAPA_TDNN(C = C).to(device)

		## Converter
		self.converter = conditionGlow(in_channel=3, n_flow=2, n_block=3).to(device)
		self.relu      = nn.ReLU()
		self.bn1       = nn.BatchNorm1d(192).to(device)
		self.bn2       = nn.BatchNorm1d(576).to(device)

		## Classifier
		self.speaker_loss       = AAMsoftmax(n_class = n_class, m = m, s = s).to(device)
		self.speaker_loss576    = AAMsoftmax576(n_class = n_class, m = m, s = s).to(device)
		self.huber              = nn.HuberLoss().to(device)


		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		para_num = sum(param.numel() for param in self.encoder_a.parameters()) * 2 + sum(param.numel() for param in self.converter.parameters())
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%((para_num) / 1024 / 1024))

	def infer_embedding(self, audio, piezo):
		if len(audio.shape) < 3:
			audio = audio.unsqueeze(1)
			piezo = piezo.unsqueeze(1)
		b, u, _ = audio.shape
		audio = audio.contiguous().view(b * u, -1)
		piezo = piezo.contiguous().view(b * u, -1)

		embeddings_audio = self.encoder_a.forward(audio)
		embeddings_piezo = self.encoder_p.forward(piezo)

		embeddings_piezo_centriods = get_centroids(embeddings_piezo.view(b, u, -1))
		embeddings_piezo_centriods_expand = embeddings_piezo_centriods.unsqueeze(1).expand(b, u, -1)

		embeddings_audio = embeddings_audio.contiguous().view(b * u, 3, 8, 8)
		embeddings_piezo_centriods_expand = embeddings_piezo_centriods_expand.contiguous().view(b * u, 3, 8, 8)

		log_p_sum, logdet, z_outs = self.converter.forward(embeddings_piezo_centriods_expand, embeddings_audio)
		z_out = self.converter.reverse(z_outs, reconstruct=True)
		# only applicable to biGlow model
		embeddings_conv = z_out.contiguous().view(b * u, -1)
		embeddings_audio = embeddings_audio.contiguous().view(b * u, -1)
		embeddings_piezo = embeddings_piezo.contiguous().view(b * u, -1)
		embeddings_cat = torch.concat([embeddings_audio, embeddings_piezo, embeddings_conv], dim=-1)
		embeddings_piezo_centriods_expand = embeddings_piezo_centriods_expand.contiguous().view(b * u, -1)

		return embeddings_audio, embeddings_piezo, self.bn1(self.relu(embeddings_conv)), self.bn2(self.relu(embeddings_cat)), embeddings_piezo_centriods_expand


	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (audio, piezo, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).to(self.device)
			b, u = labels.shape
			labels = labels.contiguous().view(b * u)
			embedding_audio, embedding_piezo, embedding_conv, embeddings_cat, embeddings_center_piezo = self.infer_embedding(audio.to(self.device), piezo.to(self.device))
			nloss_a, prec_a       = self.speaker_loss.forward(embedding_audio, labels)
			nloss_p, prec_p       = self.speaker_loss.forward(embedding_piezo, labels)	
			nloss_c, prec_c       = self.speaker_loss.forward(embedding_conv, labels)	
			nloss_cat, prec_cat   = self.speaker_loss576.forward(embeddings_cat, labels)
			nloss_huber           = self.huber(embedding_piezo, embedding_conv)


			nloss = nloss_a + nloss_p + nloss_cat + nloss_huber
			nloss.backward()
			torch.nn.utils.clip_grad_norm_(self.encoder_a.parameters(), 3.0)
			torch.nn.utils.clip_grad_norm_(self.encoder_p.parameters(), 3.0)
			torch.nn.utils.clip_grad_norm_(self.converter.parameters(), 3.0)
			self.optim.step()
			index += len(labels)
			top1 += prec_cat
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)
	
	def eval_network(self, eval_list, eval_path, eval_user, eval_uttr_enroll, eval_uttr_verify, veri_usr_lst, eval_times=10):
		EERs = []
		thresholds = []
		EER_FARs = []
		EER_FRRs = []
		for i in range(eval_times):
			EER, EER_FAR, EER_FRR, threshold = self.eval_network_one_time(eval_list, eval_path, eval_user, eval_uttr_enroll, eval_uttr_verify, veri_usr_lst)
			EERs.append(EER)
			thresholds.append(threshold)
			EER_FARs.append(EER_FAR)
			EER_FRRs.append(EER_FRR)
		minDCF, _ = ComputeMinDcf(EER_FRRs, EER_FARs, thresholds, 0.05, 1, 1)
		return np.mean(EERs), minDCF

	def eval_network_one_time(self, eval_list, eval_path, eval_user, eval_uttr_enroll, eval_uttr_verify, veri_user_list):
		self.eval()
		files = []
		embeddings = {}
		eval_dict = {}
		lines = open(eval_list).read().splitlines()
		# lines = random.sample(lines, 4000)
		for line in lines:
			id, file_path = int(line.split()[0]), eval_path + line.split()[1]
			if id in eval_dict:
				eval_dict[id].append(file_path)
			else:
				eval_dict[id] = [file_path]
		if veri_user_list == []:
			eval_users_id = random.sample(self.user_list, eval_user)
		else:
			eval_users_id = random.sample(veri_user_list, eval_user)

		total_uttr = eval_uttr_enroll + eval_uttr_verify
		audios = []
		piezos = []
		for id in eval_users_id:
			id_file_list = random.sample(eval_dict[id], total_uttr)
			for file in id_file_list:
				audio, _  = soundfile.read(os.path.join(eval_path, file))
				piezo_path = os.path.join(eval_path, file).replace('audio', 'piezo')
				piezo, _  = soundfile.read(piezo_path)
				audio = self.process_wav(audio)[0]
				piezo = self.process_wav(piezo)[0]
				audios.append(audio)
				piezos.append(piezo)

		audios = torch.from_numpy(np.array(audios)).float().to(self.device)
		piezos = torch.from_numpy(np.array(piezos)).float().to(self.device)
		embeddings_audio = self.encoder_a.forward(audios)
		embeddings_piezo = self.encoder_p.forward(piezos)
		b = eval_user
		u = eval_uttr_enroll + eval_uttr_verify
		embeddings_audio = embeddings_audio.contiguous()
		log_p_sum, logdet, z_outs = self.converter.forward(embeddings_piezo.contiguous().view(b * u, 3, 8, 8), 
													 embeddings_audio.contiguous().view(b * u, 3, 8, 8))
		z_out = self.converter.reverse(z_outs, reconstruct=True)
		# only applicable to biGlow model
		embeddings_conv = z_out.contiguous().view(b, u, -1)
		embeddings_audio = embeddings_audio.contiguous().view(b, u, -1)
		embeddings_piezo = embeddings_piezo.contiguous().view(b, u, -1)
		embeddings_cat = torch.concat([embeddings_audio, embeddings_piezo, embeddings_conv], dim=-1)
		embeddings_enroll, embeddings_verify = torch.split(embeddings_cat, [eval_uttr_enroll, eval_uttr_verify], dim=1)
		embeddings_enroll = torch.mean(embeddings_enroll, dim=1)
		sim_matrix = get_modal_cossim(embeddings_verify.contiguous(), embeddings_enroll.contiguous())
		EER, threshold, EER_FAR, EER_FRR = compute_EER(sim_matrix)

		# minDCF, _ = ComputeMinDcf(EER_FRR, EER_FAR, threshold, 0.05, 1, 1)

		return EER, EER_FAR, EER_FRR, threshold

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)
	
	def process_wav(self, audio):
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))

		audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)
		return audio