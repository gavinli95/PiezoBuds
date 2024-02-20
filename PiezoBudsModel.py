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

class PiezoBudsModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, device, **kwargs):
		super(PiezoBudsModel, self).__init__()
		self.device = device

		## Extractor
		self.encoder_a = ECAPA_TDNN(C = C).to(device)
		self.encoder_p = ECAPA_TDNN(C = C).to(device)

		## Converter
		self.converter = conditionGlow(in_channel=3, n_flow=2, n_block=3).to(device)
		self.bn = nn.BatchNorm1d(576).to(device)

		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).to(device)
		self.speaker_loss576    = AAMsoftmax576(n_class = n_class, m = m, s = s).to(device)


		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		para_num = sum(param.numel() for param in self.encoder_a.parameters()) * 2 + sum(param.numel() for param in self.converter.parameters())
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%((para_num) / 1024 / 1024))

	def infer_embedding(self, audio, piezo):
		embeddings_audio = self.encoder_a.forward(audio)
		embeddings_piezo = self.encoder_p.forward(piezo)
		
		b, f = embeddings_audio.shape

		embeddings_audio = embeddings_audio.contiguous().view(b, 3, 8, 8)
		embeddings_piezo = embeddings_piezo.contiguous().view(b, 3, 8, 8)

		log_p_sum, logdet, z_outs = self.converter.forward(embeddings_piezo, embeddings_audio)
		z_out = self.converter.reverse(z_outs, reconstruct=True)
		# only applicable to biGlow model
		embeddings_conv = z_out.contiguous().view(b, -1)
		embeddings_audio = embeddings_audio.contiguous().view(b, -1)
		embeddings_piezo = embeddings_piezo.contiguous().view(b, -1)
		embeddings_cat = torch.concat([embeddings_audio, embeddings_piezo, embeddings_conv], dim=-1)

		return embeddings_audio, embeddings_piezo, embeddings_conv, self.bn(embeddings_cat)


	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (audio, piezo, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).to(self.device)
			embedding_audio, embedding_piezo, embedding_conv, embeddings_cat = self.infer_embedding(audio.to(self.device), piezo.to(self.device))
			nloss_a, prec_a       = self.speaker_loss.forward(embedding_audio, labels)
			nloss_p, prec_p       = self.speaker_loss.forward(embedding_piezo, labels)	
			nloss_c, prec_c       = self.speaker_loss.forward(embedding_conv, labels)	
			nloss_cat, prec_cat   = self.speaker_loss576.forward(embeddings_cat, labels)

			nloss = nloss_a + nloss_p + nloss_c + nloss_cat
			nloss.backward()
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

	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		# lines = random.sample(lines, 4000)
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			piezo_path = os.path.join(eval_path, file).replace('audio', 'piezo')
			piezo, _  = soundfile.read(piezo_path)
			# Full utterance
			data_1_audio = torch.FloatTensor(numpy.stack([audio],axis=0)).to(self.device)
			data_1_piezo = torch.FloatTensor(numpy.stack([piezo],axis=0)).to(self.device)

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
				piezo = numpy.pad(piezo, (0, shortage), 'wrap')
			feats_audio = []
			feats_piezo = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats_audio.append(audio[int(asf):int(asf)+max_audio])
				feats_piezo.append(piezo[int(asf):int(asf)+max_audio])

			feats_audio = numpy.stack(feats_audio, axis = 0).astype(float)
			feats_piezo = numpy.stack(feats_piezo, axis = 0).astype(float)

			data_2_audio = torch.FloatTensor(feats_audio).to(self.device)
			data_2_piezo = torch.FloatTensor(feats_piezo).to(self.device)
			# Speaker embeddings
			with torch.no_grad():
				_, _, _, embedding_1 = self.infer_embedding(data_1_audio, data_1_piezo)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				_, _, _, embedding_2 = self.infer_embedding(data_2_audio, data_2_piezo)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF

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