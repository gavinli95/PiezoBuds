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
from sklearn.metrics.pairwise import cosine_similarity
from my_models import GE2ELoss_ori

def compute_ASR(sim_matrix, threshold):
    """
    Compute Attack Success Rate given sim matrix and threshold.

    Args:
    - sim_matrix (torch.Tensor): A similarity matrix of shape 
      (num of speakers, num of utterances, num of speakers).
    - threshold: Threshold from test 

    Returns:
    - ASR: attack success rate
    """
    num_of_speakers, num_of_utters, _ = sim_matrix.shape

    sim_matrix_thresh = sim_matrix > threshold

    # Compute ASR
    ASR = sum([(sim_matrix_thresh[i].sum() - sim_matrix_thresh[i, :, i].sum()).float()
                for i in range(num_of_speakers)]) / (num_of_speakers - 1.0) / (num_of_utters) / num_of_speakers

    # FRR = sum([(num_of_utters - sim_matrix_thresh[i, :, i].sum()).float()
    #             for i in range(num_of_speakers)]) / (num_of_utters) / num_of_speakers

    return ASR


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
		self.fc        = nn.Linear(192, 192).to(device)
		self.bn2       = nn.BatchNorm1d(576).to(device)

		## Classifier
		self.speaker_loss       = AAMsoftmax(n_class = n_class, m = m, s = s).to(device)
		self.speaker_loss576    = AAMsoftmax576(n_class = n_class, m = m, s = s).to(device)
		self.huber              = nn.HuberLoss().to(device)

		self.ge2e_a               = GE2ELoss_ori(device).to(device)
		self.ge2e_p               = GE2ELoss_ori(device).to(device)
		self.ge2e_c               = GE2ELoss_ori(device).to(device)


		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		para_num = sum(param.numel() for param in self.encoder_a.parameters()) * 2 + sum(param.numel() for param in self.converter.parameters())
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%((para_num) / 1024 / 1024))

	def infer_embedding(self, audio, piezo, labels):
		audio = audio.float()
		piezo = piezo.float()
		if len(audio.shape) < 3:
			audio = audio.unsqueeze(1)
			piezo = piezo.unsqueeze(1)
		b, u, _ = audio.shape
		audio = audio.contiguous().view(b * u, -1)
		piezo = piezo.contiguous().view(b * u, -1)

		# if labels != None:
		# 	tmp_labesl = labels.detach().cpu().numpy()
		# 	tmp_audio = audio.detach().cpu().numpy()
		# 	tmp_piezo = piezo.detach().cpu().numpy()

		# 	for i in range(5):
		# 		plt.figure()
		# 		plt.plot(np.abs(tmp_piezo[i]))
		# 		# tmp = square_envelope(np.abs(tmp_piezo[i]), segment_length=1000)
		# 		tmp = extract_envelope(np.abs(tmp_piezo[i]), kernel_size=101)
		# 		plt.plot(tmp)
		# 		plt.savefig(f'0_piezo_{i}_{labels[i]}.png')
		# 		plt.close()

		# 		plt.figure()
		# 		plt.plot(tmp_audio[i])
		# 		ev = tmp / np.max(np.abs(tmp))
		# 		ev[ev > 0.21] = 1
		# 		ev[ev <= 0.21] = 1e-6
		# 		plt.plot(ev)
		# 		plt.savefig(f'0_audio_{i}_{labels[i]}.png')
		# 		plt.close()

		# if labels != None:
		# 	tmp_labesl = labels.detach().cpu().numpy()
		# 	# tmp_audio = audio.detach().cpu().numpy()
		# 	# tmp_piezo = piezo.detach().cpu().numpy()

		# 	tmp_audio = self.encoder_a.specaug_total(audio)
		# 	tmp_piezo = self.encoder_p.specaug_total(piezo)
		# 	tmp_audio = tmp_audio.detach().cpu().numpy()
		# 	tmp_piezo = tmp_piezo.detach().cpu().numpy()

		# 	for i in range(5):
		# 		plt.figure()
		# 		plt.imshow(tmp_audio[i], cmap='viridis')
		# 		plt.colorbar()
		# 		plt.savefig(f'audio_{i}_{labels[i]}.png')
		# 		plt.close()

		# 		plt.figure()
		# 		plt.imshow(tmp_piezo[i], cmap='viridis')
		# 		plt.colorbar()
		# 		plt.savefig(f'piezo_{i}_{labels[i]}.png')
		# 		plt.close()


		embeddings_audio = self.encoder_a.forward(audio)
		embeddings_piezo = self.encoder_p.forward(piezo)

		embeddings_piezo_centriods = get_centroids(embeddings_piezo.view(b, u, -1))
		# embeddings_piezo_centriods_expand = embeddings_piezo_centriods.unsqueeze(1).expand(b, u, -1)

		embeddings_audio = embeddings_audio.contiguous().view(b * u, 3, 8, 8)
		# embeddings_piezo_centriods_expand = embeddings_piezo_centriods_expand.contiguous().view(b * u, 3, 8, 8)
		embeddings_piezo = embeddings_piezo.contiguous().view(b * u, 3, 8, 8)

		log_p_sum, logdet, z_outs = self.converter.forward(embeddings_piezo, embeddings_audio)
		z_out = self.converter.reverse(z_outs, reconstruct=True)
		
		embeddings_conv = z_out.contiguous().view(b * u, -1)
		# embeddings_conv = self.bn1(self.fc(embeddings_conv))

		embeddings_audio = embeddings_audio.contiguous().view(b * u, -1)
		embeddings_piezo = embeddings_piezo.contiguous().view(b * u, -1)
		embeddings_cat = torch.concat([embeddings_audio, embeddings_piezo, embeddings_conv], dim=-1)
		#embeddings_piezo_centriods_expand = embeddings_piezo_centriods_expand.contiguous().view(b * u, -1)

		# return embeddings_audio, embeddings_piezo, self.bn1(self.relu(embeddings_conv)), self.bn2(self.relu(embeddings_cat)), embeddings_piezo_centriods_expand
		return embeddings_audio, embeddings_piezo, embeddings_conv, F.normalize(embeddings_cat, p=2, dim=-1)



	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (audio, piezo, audio_extra, noise, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).to(self.device)
			b, u = labels.shape
			labels = labels.contiguous().view(b * u)
			embedding_audio, embedding_piezo, embedding_conv, embeddings_cat = self.infer_embedding(audio.to(self.device), piezo.to(self.device), labels)
			_, _, non_concurrent_conv, _ = self.infer_embedding(audio_extra.to(self.device), piezo.to(self.device), None)
			_, _, fake_conv_audio_audio, _ = self.infer_embedding(audio.to(self.device), audio.to(self.device), None)
			_, embedding_white_noise, fake_conv_audio_white, _ = self.infer_embedding(audio.to(self.device), noise.to(self.device).float(), None)
			nloss_a, prec_a       = self.speaker_loss.forward(embedding_audio, labels)
			nloss_p, prec_p       = self.speaker_loss.forward(embedding_piezo, labels)	
			# nloss_c, prec_c       = self.speaker_loss.forward(embedding_conv, labels)	
			# nloss_cat, prec_cat   = self.speaker_loss576.forward(embeddings_cat, labels)
			# nloss_huber           = self.huber(embedding_piezo, embedding_conv)
			# nloss_a = self.ge2e_a.forward(embedding_audio.contiguous().view(b, u, -1))
			
			# nloss_p = self.ge2e_p.forward(embedding_piezo.contiguous().view(b, u, -1))
			# nloss_z = self.ge2e_c.forward(embedding_conv.contiguous().view(b, u, -1))
			embedding_piezo = embedding_piezo.contiguous().view(b, u, -1)
			embedding_conv = embedding_conv.contiguous().view(b, u, -1)
			embedding_audio = embedding_audio.contiguous().view(b, u, -1)
			non_concurrent_conv = non_concurrent_conv.contiguous().view(b, u, -1)
			fake_conv_audio_audio = fake_conv_audio_audio.contiguous().view(b, u, -1)
			fake_conv_audio_white = fake_conv_audio_white.contiguous().view(b, u, -1)

			# nloss_apz = self.ge2e_a.forward(torch.concat(
			# 	[# embedding_piezo, 
	 		# 	 embedding_conv,
			# 	 embedding_audio,
			# 	 ], dim=0
			# ))

			nloss_apzff = self.ge2e_c.forward(torch.concat(
				[
				 # embedding_audio,
				 # embedding_piezo,
	 			 embedding_conv,
				 non_concurrent_conv,
				 fake_conv_audio_audio,
				 fake_conv_audio_white
				], dim=0
			))

			# nloss_defense = 0.0
			# for i in range(b):
			# 	nloss_defense += self.ge2e_c.forward(torch.concat([
			# 		embedding_conv[i].unsqueeze(0),
			# 		fake_conv_audio_audio[i].unsqueeze(0),
			# 		fake_conv_audio_white[i].unsqueeze(0)
			# 	], dim=0))


			# nloss = nloss_a + nloss_p + nloss_cat + nloss_huber
			nloss = nloss_a + nloss_p + nloss_apzff
			nloss.backward()
			torch.nn.utils.clip_grad_norm_(self.encoder_a.parameters(), 3.0)
			torch.nn.utils.clip_grad_norm_(self.encoder_p.parameters(), 3.0)
			torch.nn.utils.clip_grad_norm_(self.converter.parameters(), 3.0)
			torch.nn.utils.clip_grad_norm_(self.ge2e_a.parameters(), 3.0)
			torch.nn.utils.clip_grad_norm_(self.ge2e_p.parameters(), 3.0)
			torch.nn.utils.clip_grad_norm_(self.ge2e_c.parameters(), 3.0)
			self.optim.step()
			index += len(labels)
			# top1 += prec_cat
			top1 += 0
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)
	
	def eval_network(self, eval_list, eval_path, eval_user_total,
				  eval_user, eval_uttr_enroll, eval_uttr_verify, veri_usr_lst,
				  eval_noise_type, eval_noise_path, 
				  eval_motion_type, eval_motion_path,
				  eval_times=10):
		EERs = []
		thresholds = []
		EER_FARs = []
		EER_FRRs = []
		FAR_replays = []
		EER_audios = []
		EER_piezos = []
		thres_audios = []
		thres_piezos = []
		thres_convs = []
		users = [i for i in range(81)]
		users_list = random.sample(users, eval_user_total)
		for i in range(eval_times):
			EER, EER_FAR, EER_FRR, threshold, FAR_replay, EER_audio, EER_piezo, thres_audio, thres_piezo, thres_conv = self.eval_network_one_time(eval_list, eval_path, eval_user, eval_uttr_enroll, eval_uttr_verify, users_list,
																			 eval_noise_type, eval_noise_path, 
				  															 eval_motion_type, eval_motion_path)
			EERs.append(EER)
			thresholds.append(threshold)
			EER_FARs.append(EER_FAR)
			EER_FRRs.append(EER_FRR)
			FAR_replays.append(FAR_replay.item())
			EER_audios.append(EER_audio)
			EER_piezos.append(EER_piezo)
			thres_audios.append(thres_audio)
			thres_piezos.append(thres_piezo)
			thres_convs.append(thres_conv)
		minDCF, _ = ComputeMinDcf(EER_FRRs, EER_FARs, thresholds, 0.05, 1, 1)
		return np.mean(EERs), minDCF, np.mean(thresholds), np.mean(FAR_replays), np.mean(EER_audios), np.mean(EER_piezos), np.mean(thres_audios), np.mean(thres_piezos), np.mean(thres_convs)
	
	def get_embeddings_cat(self, u, audios, piezos):
		audios = torch.from_numpy(np.array(audios)).float().to(self.device)
		piezos = torch.from_numpy(np.array(piezos)).float().to(self.device)
		
		embeddings_audio = self.encoder_a.forward(audios)
		del audios
		torch.cuda.empty_cache()
		embeddings_piezo = self.encoder_p.forward(piezos)
		del piezos
		torch.cuda.empty_cache()
		embeddings_audio = embeddings_audio.contiguous()
		log_p_sum, logdet, z_outs = self.converter.forward(embeddings_piezo.contiguous().view(u, 3, 8, 8), 
													embeddings_audio.contiguous().view(u, 3, 8, 8))
		z_out = self.converter.reverse(z_outs, reconstruct=True)
		# only applicable to biGlow model
		embeddings_conv = z_out.contiguous().view(u, -1)
		# embeddings_conv = self.bn1(self.fc(embeddings_conv))
		embeddings_conv = embeddings_conv.contiguous().view(1, u, -1)

		embeddings_audio = embeddings_audio.contiguous().view(1, u, -1)
		embeddings_piezo = embeddings_piezo.contiguous().view(1, u, -1)
		embeddings_cat = torch.concat([embeddings_audio, embeddings_piezo, embeddings_conv], dim=-1)
		# embeddings_cat = self.fc(embeddings_cat)
		embeddings_cat = F.normalize(embeddings_cat, p=2, dim=-1)
		return embeddings_cat, embeddings_conv, embeddings_audio, embeddings_piezo

	def eval_network_one_time(self, eval_list, eval_path, eval_user, eval_uttr_enroll, eval_uttr_verify, veri_user_list,
						   		eval_noise_type, eval_noise_path, 
				  				eval_motion_type, eval_motion_path):
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

		embeddings_cat = []
		embeddings_conv = []
		embeddings_audio = []
		embeddings_piezo = []

		embeddings_cat_replayattack = [] # piezo will be white noise
		embeddings_conv_replayattack = []
		embeddings_audio_replayattack = []
		embeddings_piezo_replayattack = []

		total_uttr = eval_uttr_enroll + eval_uttr_verify
		b = eval_user
		u = eval_uttr_enroll + eval_uttr_verify
		
		for id in eval_users_id:
			# data for evaluating network
			audios = []
			piezos = []
			noises = []
			motions = []

			id_file_list = random.sample(eval_dict[id], total_uttr)
			for file in id_file_list:
				# data for evaluating network
				audio, _  = soundfile.read(os.path.join(eval_path, file))
				piezo_path = os.path.join(eval_path, file).replace('audio', 'piezo')
				piezo, _  = soundfile.read(piezo_path)
				audio = self.process_wav(audio)[0]
				piezo = self.process_wav(piezo)[0]

				audios.append(audio)
				piezos.append(piezo)

				if eval_noise_type != 0:
					noise_types = ['', 'whitenoise', 'conversation', 'cafe', 'restaurant', 'construction']
					noise_file = os.path.join(eval_noise_path, noise_types[eval_noise_type] + '.wav')
					noise, _ = soundfile.read(noise_file)
					noise = self.process_wav(noise)[0]
					noises.append(noise)
				if eval_motion_type != 0:
					motion_types = ['', 'turn', 'tap', 'clap', 'walk']
					motion_file = os.path.join(eval_motion_path, motion_types[eval_motion_type] + '.wav')
					motion, _ = soundfile.read(motion_file)
					motion = self.process_wav(motion)[0]
					motions.append(motion)

			if eval_noise_type != 0:
				# audios = (2 * np.array(audios) + np.array(noises)) / 3
				noises = np.array(noises) / np.max(np.abs(noises))
				audios = self.add_noise(np.array(audios), noises, 0, 1)
			if eval_motion_type != 0:
				motions = np.array(motions)
				motions = motions / np.max(np.abs(motions))
				# piezos = (np.array(piezos) + np.array(motions) * 0.5) / 2
				piezos = self.add_noise(np.array(piezos), np.array(motions), 0, 1)
				
			embedding_cat, embedding_conv, embedding_audio, embedding_piezo = self.get_embeddings_cat(u, audios, piezos)
			embeddings_cat.append(embedding_cat)
			embeddings_conv.append(embedding_conv)
			embeddings_audio.append(embedding_audio)
			embeddings_piezo.append(embedding_piezo)

			# data for evaluating replay
			audios = []
			piezos = []
			id_file_list = random.sample(eval_dict[id], eval_uttr_verify * 2)
			id_file_list_replay_audio = id_file_list[0: eval_uttr_verify]
			id_file_list = id_file_list[eval_uttr_verify: ]
			for i in range(eval_uttr_verify):
				file = id_file_list[i]
				file_replay_audio = id_file_list_replay_audio[i]
				# add loading audio from attack set later
				
				# replay attack while user is silent
				# leave piezo as white noise
				# audio, _  = soundfile.read(os.path.join(eval_path, file))
				# audio = self.process_wav(audio)[0]
				# piezo, sr = soundfile.read("./noise_piezo.wav")
				# piezo = self.process_wav(piezo)[0]
				# env = extract_envelope(np.abs(piezo))
				# env = env / np.max(np.abs(env))
				# if 1 - np.min(env[1000: len(env)-1000]) < 0.45:
				# 	env = env * 0
				# env = self.process_wav(env)[0]
				# audios.append(audio * env)
				# piezos.append(piezo)
				# piezos.append(audio)
	
				# test replay attack while talking
				audio, _  = soundfile.read(os.path.join(eval_path, file))
				piezo_path = os.path.join(eval_path, file).replace('audio', 'piezo')
				piezo, _  = soundfile.read(piezo_path)
				audio_replay, _ = soundfile.read(os.path.join(eval_path, file_replay_audio))
				audio = self.process_wav(audio)[0]
				audio_replay = self.process_wav(audio_replay)[0]
				audio = audio + audio_replay
				audio = audio / np.max(np.abs(audio))
				piezo = self.process_wav(piezo)[0]
				audios.append(audio)
				piezos.append(audio)

			embedding_cat, embedding_conv, embedding_audio, embedding_piezo = self.get_embeddings_cat(eval_uttr_verify, audios, piezos)
			embeddings_cat_replayattack.append(embedding_cat)
			embeddings_conv_replayattack.append(embedding_conv)
			embeddings_audio_replayattack.append(embedding_audio)
			embeddings_piezo_replayattack.append(embedding_piezo)

		# evaluate performance
		embeddings_cat = torch.concat(embeddings_cat, dim=0)
		embeddings_conv = torch.concat(embeddings_conv, dim=0)
		embeddings_audio = torch.concat(embeddings_audio, dim=0)
		embeddings_piezo = torch.concat(embeddings_piezo, dim=0)

		# embeddings_enroll, embeddings_verify = torch.split(embeddings_cat, [eval_uttr_enroll, eval_uttr_verify], dim=1)
		# embeddings_enroll = torch.mean(embeddings_enroll, dim=1) # centroids for users
		# sim_matrix = get_modal_cossim_revised(embeddings_verify.contiguous(), embeddings_enroll.contiguous())
		# EER, threshold, EER_FAR, EER_FRR = compute_EER(sim_matrix)

		embeddings_enroll_conv, embeddings_verify_conv = torch.split(embeddings_conv, [eval_uttr_enroll, eval_uttr_verify], dim=1)
		embeddings_enroll_conv = torch.mean(embeddings_enroll_conv, dim=1) # centroids for users
		sim_matrix_conv = get_modal_cossim_revised(embeddings_verify_conv.contiguous(), embeddings_enroll_conv.contiguous())
		_, thres_conv, _, _, = compute_EER(sim_matrix_conv)

		embeddings_enroll_audio, embeddings_verify_audio = torch.split(embeddings_audio, [eval_uttr_enroll, eval_uttr_verify], dim=1)
		embeddings_enroll_audio = torch.mean(embeddings_enroll_audio, dim=1) # centroids for users
		sim_matrix_audio = get_modal_cossim_revised(embeddings_verify_audio.contiguous(), embeddings_enroll_audio.contiguous())
		EER_audio, thres_audio, _, _, = compute_EER(sim_matrix_audio)

		embeddings_enroll_piezo, embeddings_verify_piezo = torch.split(embeddings_piezo, [eval_uttr_enroll, eval_uttr_verify], dim=1)
		embeddings_enroll_piezo = torch.mean(embeddings_enroll_piezo, dim=1) # centroids for users
		sim_matrix_piezo = get_modal_cossim_revised(embeddings_verify_piezo.contiguous(), embeddings_enroll_piezo.contiguous())
		EER_piezo, thres_piezo, _, _, = compute_EER(sim_matrix_piezo)
		# sim_matrix = get_modal_cossim_revised(embeddings_verify_conv.contiguous(), embeddings_enroll_piezo.contiguous())

		sim_matrix = (1 * sim_matrix_conv + sim_matrix_audio + 1 * sim_matrix_piezo) / 3
		EER, threshold, EER_FAR, EER_FRR = compute_EER(sim_matrix, threshold=0.56147)

		thres_audio = 0.5190
		thres_piezo = 0.60798
		thres_conv = 0.68924
		# sim_matrix_thresh = (sim_matrix_piezo > (thres_piezo)) & (sim_matrix_conv > (thres_conv)) & (sim_matrix_audio > thres_audio)
		# num_of_speakers, num_of_utters, _ = sim_matrix_thresh.shape
		# FAR = sum([(sim_matrix_thresh[i].sum() - sim_matrix_thresh[i, :, i].sum()).float()
        #                 for i in range(num_of_speakers)]) / (num_of_speakers - 1.0) / (num_of_utters) / num_of_speakers

		# FRR = sum([(num_of_utters - sim_matrix_thresh[i, :, i].sum()).float()
        #             for i in range(num_of_speakers)]) / (num_of_utters) / num_of_speakers
		
		# EER = (FAR + FRR) / 2
		# EER = EER.item()

		

		# evaluate replay attack
		# embeddings_cat_replayattack = torch.concat(embeddings_cat_replayattack, dim=0)
		# sim_matrix = get_modal_cossim_revised(embeddings_cat_replayattack.contiguous(), embeddings_enroll.contiguous())
		# sim_matrix_thresh = sim_matrix > threshold
		embeddings_conv_replayattack = torch.concat(embeddings_conv_replayattack, dim=0)
		sim_matrix_conv = get_modal_cossim_revised(embeddings_conv_replayattack.contiguous(), embeddings_enroll_conv.contiguous())
		np.savetxt('sim_matrix_z.txt', sim_matrix_conv.detach().cpu().numpy()[0, :, 0])

		embeddings_audio_replayattack = torch.concat(embeddings_audio_replayattack, dim=0)
		sim_matrix_audio = get_modal_cossim_revised(embeddings_audio_replayattack.contiguous(), embeddings_enroll_audio.contiguous())
		np.savetxt('sim_matrix_a.txt', sim_matrix_audio.detach().cpu().numpy()[0, :, 0])

		embeddings_piezo_replayattack = torch.concat(embeddings_piezo_replayattack, dim=0)
		sim_matrix_piezo = get_modal_cossim_revised(embeddings_piezo_replayattack.contiguous(), embeddings_enroll_piezo.contiguous())
		np.savetxt('sim_matrix_p.txt', sim_matrix_piezo.detach().cpu().numpy()[0, :, 0])
		
		# sim_matrix = (1 * sim_matrix_conv + 0 * sim_matrix_audio + 1 * sim_matrix_piezo) / 2
		# sim_matrix = get_modal_cossim_revised(embeddings_conv_replayattack.contiguous(), embeddings_enroll_piezo.contiguous())
		sim_matrix_thresh = (sim_matrix_piezo > (thres_piezo)) & (sim_matrix_conv > (thres_conv)) & (sim_matrix_audio > thres_audio)
		FAR_replay = sum([(sim_matrix_thresh[i, :, i].sum()).float()
                   for i in range(b)]) / (eval_uttr_verify * b)
		# FAR_replay = compute_ASR(sim_matrix, threshold)
		

		return EER, EER_FAR, EER_FRR, threshold, FAR_replay, EER_audio, EER_piezo, thres_audio, thres_piezo, thres_conv


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
	
	def add_noise(self, audio, noise, low, high):
		clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		noise_db = 10 * numpy.log10(numpy.mean(noise ** 2)+1e-4)
		noisesnr = random.uniform(low, high)
		noise = numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noise
		return audio + 0.1 * noise

