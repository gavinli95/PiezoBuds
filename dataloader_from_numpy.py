import torch
import itertools
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from random import shuffle
from utils import *
import scipy


def find_all_files(directory, type):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(type):
                file_paths.append(os.path.join(root, file))
    return file_paths


class LatentFeatureSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.from_numpy(self.features[idx]).float()
        labels = torch.tensor(self.labels[idx])
        return features, labels


class FrequencyFeatureSet(Dataset):
    def __init__(self, shuffle, user_ids, m):
        self.shuffle = shuffle
        self.user_ids = user_ids
        self.m = m

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        # if self.shuffle:
        #     id = random.sample(self.user_ids, 1)[0]  # select random speaker
        # else:

        id = self.user_ids[idx]

        root_piezo = './frequencybins/{}_piezo.npy'.format(id)
        root_audio = './frequencybins/{}_audio.npy'.format(id)
        root_noise = './frequencybins/{}_noise.npy'.format(id)

        piezo_features = np.load(root_piezo)
        audio_features = np.load(root_audio)
        noise_features = np.load(root_noise)
        piezo_audio_features = np.concatenate((piezo_features, audio_features), axis=1)

        noise_audio_features = np.concatenate((noise_features, audio_features), axis=1)

        utter_index = random.sample(range(piezo_audio_features.shape[0]), self.m)  # select M utterances per speaker
        utterance = piezo_audio_features[utter_index]

        noises = noise_audio_features[utter_index]

        utterance = torch.from_numpy(utterance).float()
        noises = torch.from_numpy(noises).float()

        return utterance, noises


class STFTFeatureSet(Dataset):
    def __init__(self, shuffle, user_ids, m, file_dir):
        self.shuffle = shuffle
        self.user_ids = user_ids
        self.m = m
        self.file_dir = file_dir
        if self.shuffle:
            random.shuffle(self.user_ids)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        # if self.shuffle:
        #     id = random.sample(self.user_ids, 1)[0]  # select random speaker
        # else:
        id = self.user_ids[idx]

        root_piezo = self.file_dir + '{}_piezo.npy'.format(id)
        root_audio = self.file_dir + '{}_audio.npy'.format(id)
        # root_noise = './stft/{}_noise.npy'.format(id)

        piezo_features = np.load(root_piezo)
        audio_features = np.load(root_audio)
        # noise_features = np.load(root_noise)
        piezo_audio_features = np.concatenate((piezo_features, audio_features), axis=1)

        # noise_audio_features = np.concatenate((noise_features, audio_features), axis=1)

        utter_index = random.sample(range(piezo_audio_features.shape[0]), self.m)  # select M utterances per speaker
        utterance = piezo_audio_features[utter_index]

        # noises = noise_audio_features[utter_index]

        utterance = torch.from_numpy(utterance).float()
        # noises = torch.from_numpy(noises).float()

        return utterance, id

class VoxSTFTFeatureSet(Dataset):
    def __init__(self, shuffle, user_ids, m, file_dir):
        self.shuffle = shuffle
        self.user_ids = user_ids
        self.m = m
        self.file_dir = file_dir
        if self.shuffle:
            random.shuffle(self.user_ids)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        # if self.shuffle:
        #     id = random.sample(self.user_ids, 1)[0]  # select random speaker
        # else:
        id = self.user_ids[idx]
        root_audio = self.file_dir + '{}_audio.npy'.format(id)

        audio_features = np.load(root_audio)
        
        utter_index = random.sample(range(audio_features.shape[0]), self.m)  # select M utterances per speaker
        utterance = audio_features[utter_index]

        utterance = torch.from_numpy(utterance).float()
        
        return utterance, id
    

class STFTFeatureSet4Class(Dataset):
    def __init__(self, n_user, train_ratio, time_stamp, train):
        self.n_user = n_user
        self.train_ratio = train_ratio
        self.time_stamp = time_stamp
        self.train = train
        # load the npy file to get the len of the overall training data
        self.file_pth = './training_data/' + str(self.n_user) + '/' + str(int(self.train_ratio * 100)) + '/' + time_stamp
        if self.train:
            self.data_lst = np.load(self.file_pth + '/train_n_lst.npy')
        else:
            self.data_lst = np.load(self.file_pth + '/test_n_lst.npy')
        
        self.n_data = len(self.data_lst)
        if self.train:
            self.pa_data = np.load(self.file_pth + '/train_data.npy')
        else:
            self.pa_data = np.load(self.file_pth + '/test_data.npy')
        

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):

        utterance = self.pa_data[idx] # get the corresponding utterance profile

        utterance = torch.from_numpy(utterance).float()

        user_id = self.data_lst[idx]

        user_id = torch.tensor(user_id)

        return utterance, user_id


class VoxcelebSet4Class(Dataset):
    def __init__(self, n_user, dir_root, is_audio):
        self.n_user = n_user
        self.root = dir_root
        self.is_audio = is_audio

        total_n_samples = 0
        total_labels = []
        for i in range(n_user):
            # if i == 39:
            #     continue
            user_i_dir = dir_root + '{}/'.format(i)
            files = find_all_files(user_i_dir, '.npy')
            n_samples = len(files)
            total_n_samples += n_samples
            total_labels += [(i, _) for _ in range(n_samples)]
        self.total_n_samples = total_n_samples
        self.total_labels = total_labels
        


    def __len__(self):
        return self.total_n_samples

    def __getitem__(self, idx):
        (i, uttr) = self.total_labels[idx]
        if self.is_audio:
            pth = self.root + '{}/{}_{}_audio.npy'.format(i, i, uttr)
        else:
            pth = self.root + '{}/{}_{}_piezo.npy'.format(i, i, uttr)
        utterance = np.load(pth)
        utterance = torch.from_numpy(utterance).float()
        i = torch.tensor(i)
        return utterance, i


class WavDataset(Dataset):
    # dataset_dir example
    # ./processed_data/wav_clips/piezobuds/
    # if is_multi_moda, return (piezo, audio, id)
    # else if is_audio, return (audio, id)
    # else return (piezo, id)
    def __init__(self, dataset_dir, n_user, is_multi_moda=True, is_audio=True):
        super().__init__()

        self.is_audio = is_audio
        self.is_multi_moda = is_multi_moda
        self.n_user = n_user

        self.dir_piezo = dataset_dir + 'piezo/'
        self.dir_audio = dataset_dir + 'audio/'

        total_n_samples = 0
        total_labels = []
        for i in range(n_user):
            user_i_dir = self.dir_audio + '{}/'.format(i)
            files = find_all_files(user_i_dir, '.wav')
            n_samples = len(files)
            total_n_samples += n_samples
            total_labels += [(i, _) for _ in range(n_samples)]
        self.total_n_samples = total_n_samples
        self.total_labels = total_labels

    def __len__(self):
        return self.total_n_samples
    
    def __getitem__(self, idx):
        (i, wav_idx) = self.total_labels[idx]
        if self.is_multi_moda:
            _, piezo = scipy.io.wavfile.read(self.dir_piezo + '{}/{}.wav'.format(i, wav_idx))
            _, audio = scipy.io.wavfile.read(self.dir_audio + '{}/{}.wav'.format(i, wav_idx))
            piezo = torch.from_numpy(piezo).float()
            audio = torch.from_numpy(audio).float()
            return (piezo, audio, i)
        else:
            if self.is_audio:
                _, audio = scipy.io.wavfile.read(self.dir_audio + '{}/{}.wav'.format(i, wav_idx))
                audio = torch.from_numpy(audio).float()
                return (audio.float(), i)
            else:
                _, piezo = scipy.io.wavfile.read(self.dir_piezo + '{}/{}.wav'.format(i, wav_idx))
                piezo = torch.from_numpy(piezo).float()
                return (piezo.float(), i)
        

class WavDatasetForVerification(Dataset):
    # dataset_dir example
    # ./processed_data/wav_clips/piezobuds/
    # if is_multi_moda, return (piezo, audio, id)
    # else if is_audio, return (audio, id)
    # else return (piezo, id)
    def __init__(self, dataset_dir, n_user_list, m, is_multi_moda=True, is_audio=True):
        super().__init__()

        self.is_audio = is_audio
        self.is_multi_moda = is_multi_moda
        self.n_user_list = n_user_list
        self.m = m

        self.dir_piezo = dataset_dir + 'piezo/'
        self.dir_audio = dataset_dir + 'audio/'

    def __len__(self):
        return len(self.n_user_list)
    
    def __getitem__(self, idx):
        user = self.n_user_list[idx]

        if self.is_multi_moda:
            piezo_root = self.dir_piezo + '{}/'.format(user)
            audio_root = self.dir_audio + '{}/'.format(user)

            file_list = find_all_files(audio_root, '.wav')
            num_utter = len(file_list)
            samples_idx = random.sample(list(range(num_utter)), self.m)

            piezos = []
            audios = []
            ids = []
            for sid in samples_idx:
                _, piezo = scipy.io.wavfile.read(piezo_root + '{}.wav'.format(sid))
                _, audio = scipy.io.wavfile.read(audio_root + '{}.wav'.format(sid))
                piezos.append(piezo)
                audios.append(audio)
                ids.append(user)
            piezos = np.array(piezos)
            audios = np.array(audios)
            ids = np.array(ids)
            piezos = torch.from_numpy(piezos).float()
            audios = torch.from_numpy(audios).float()
            ids = torch.from_numpy(ids)


            return (piezos, audios, ids)

