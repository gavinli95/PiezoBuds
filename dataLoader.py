'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal

class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, num_uttr, eval_user_total, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		# Load and configure augmentation files
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		# Load data & labels
		self.data_list  = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name     = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)
		self.data_dict = {}
		maxid = 0
		for i in range(len(self.data_label)):
			id = self.data_label[i]
			if id > maxid:
				maxid = id
			if id in self.data_dict:
				self.data_dict[id].append(self.data_list[i])
			else:
				self.data_dict[id] = [self.data_list[i]]
		number_of_train_users = maxid - eval_user_total + 1

		usr_list = [i for i in range(maxid + 1)]
		self.user_list_train = random.sample(usr_list, number_of_train_users)
		self.user_list_veri  = [i for i in usr_list if i not in self.user_list_train]
		self.num_uttr = num_uttr

	def return_user_lists(self):
		return self.user_list_train, self.user_list_veri

	def process_wav(self, audio):
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))

		audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)
		return audio

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		userid = self.user_list_train[index]

		audios = []
		piezos = []
		ids = []

		file_paths = random.sample(self.data_dict[userid], self.num_uttr)
		for file in file_paths:
			audio, sr = soundfile.read(file)
			piezo_path = file.replace("audio", "piezo")
			piezo, sr = soundfile.read(piezo_path)		

			audio = self.process_wav(audio)
			piezo = self.process_wav(piezo)
			audios.append(audio[0])
			piezos.append(piezo[0])
			ids.append(userid)
		
		# length = self.num_frames * 160 + 240
		# if audio.shape[0] <= length:
		# 	shortage = length - audio.shape[0]
		# 	audio = numpy.pad(audio, (0, shortage), 'wrap')
		# 	piezo = numpy.pad(piezo, (0, shortage), 'wrap')
		# start_frame = numpy.int64(random.random()*(audio.shape[0]-length))

		# audio = audio[start_frame:start_frame + length]
		# audio = numpy.stack([audio],axis=0)

		# piezo = piezo[start_frame:start_frame + length]
		# piezo = numpy.stack([piezo],axis=0)

		# Data Augmentation
		# augtype = random.randint(0,5)
		# augtype = 0
		# if augtype == 0:   # Original
		# 	audio = audio
		# elif augtype == 1: # Reverberation
		# 	audio = self.add_rev(audio)
		# elif augtype == 2: # Babble
		# 	audio = self.add_noise(audio, 'speech')
		# elif augtype == 3: # Music
		# 	audio = self.add_noise(audio, 'music')
		# elif augtype == 4: # Noise
		# 	audio = self.add_noise(audio, 'noise')
		# elif augtype == 5: # Television noise
		# 	audio = self.add_noise(audio, 'speech')
		# 	audio = self.add_noise(audio, 'music')
		audios = numpy.array(audios).astype(float)
		piezos = numpy.array(piezos).astype(float)
		ids = numpy.array(ids).astype(int)
		return torch.from_numpy(audios).float(), torch.from_numpy(piezos).float(), torch.from_numpy(ids)

	def __len__(self):
		return len(self.user_list_train)

	def add_rev(self, audio):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio