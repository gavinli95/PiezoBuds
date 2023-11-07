#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:56:19 2018

@author: harry
"""
import os
import random
import librosa
import torch
import torch.autograd as grad
import torch.nn.functional as F

# from hparam import hparam as hp

import numpy as np
import networkx as nx
from netgraph import Graph
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans


def draw_distance_matrix(distance_matrix, filename):
    edges = []
    n = distance_matrix.shape[0]
    for i in [0, 16]:
        for j in range(n):
            if i != j:
                edges.append((i, j, {'weight': distance_matrix[i, j]}))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, weight='weight')
    colors = []
    colors.append('red')
    for i in range(0, 15):
        colors.append('red')
    colors.append('blue')
    for i in range(0, 15):
        colors.append('blue')
    alphas = []
    alphas.append(1)
    for i in range(0, 15):
        alphas.append(0.5)
    alphas.append(1)
    for i in range(0, 15):
        alphas.append(0.5)
    nx.draw(G, pos, with_labels=False, node_color=colors, node_size=100, alpha=alphas)
    plt.savefig(filename)
    # plt.show()


def get_distance_matrix(embeddings):
    n = embeddings.shape[0]
    result = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            result[i][j] = 1 - cosine_similarity(embeddings[i], embeddings[j])
    return result

def get_centroids_prior(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid/len(speaker)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids

def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids

def get_centroids_kmeans(embeddings):
    num_of_user = embeddings.shape[0]
    centroids = []
    for i in range(num_of_user):
        X = embeddings[i].cpu()
        num_clusters = 1
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        cluster_center = kmeans.cluster_centers_[0]
        centroids.append(cluster_center)
    return centroids

def get_centroid(embeddings, speaker_num, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid/(len(embeddings[speaker_num])-1)
    return centroid

def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance

    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids

def get_cossim_prior(embeddings, centroids):
    # Calculates cosine similarity matrix. Requires (N, M, feature) input
    cossim = torch.zeros(embeddings.size(0),embeddings.size(1),centroids.size(0))
    for speaker_num, speaker in enumerate(embeddings):
        for utterance_num, utterance in enumerate(speaker):
            for centroid_num, centroid in enumerate(centroids):
                if speaker_num == centroid_num:
                    centroid = get_centroid(embeddings, speaker_num, utterance_num)
                output = F.cosine_similarity(utterance,centroid,dim=0)+1e-6
                cossim[speaker_num][utterance_num][centroid_num] = output
    return cossim


def cosine_similarity(tensor1, tensor2):
    """
    Calculate cosine similarity between two 1D tensors.

    Args:
        tensor1 (torch.Tensor): First input tensor.
        tensor2 (torch.Tensor): Second input tensor.

    Returns:
        torch.Tensor: Cosine similarity between the two input tensors.
    """
    # Normalize the tensors
    tensor1 = F.normalize(tensor1, dim=0)
    tensor2 = F.normalize(tensor2, dim=0)
    tensor1 = torch.reshape(tensor1, (1, 256))
    tensor2 = torch.reshape(tensor2, (1, 256))

    # Calculate cosine similarity
    similarity = F.cosine_similarity(tensor1, tensor2)

    return similarity

def get_cossim(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1],
        -1
    )
    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances,
        -1
    )
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    )
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff

def get_modal_cossim(embeddings, centroids):
    # only calculate the cosine similarities between the different modalities instead of itself
    num_user = embeddings.shape[0]
    num_utterances = embeddings.shape[1]

    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances,
        -1
    )

    embeddings_expand = embeddings_flat.unsqueeze(1).repeat((1, num_user, 1))
    embeddings_expand = embeddings_expand.view(embeddings_expand.shape[0] * embeddings_expand.shape[1], -1)

    centriods_expand = centroids.repeat((num_utterances * num_user, 1))

    # calculate the consine similarites between embedding vectors and centroids
    cossim_modal = F.cosine_similarity(embeddings_expand, centriods_expand)
    cossim_modal = cossim_modal.view(num_user, num_utterances, -1)
    cossim_modal = cossim_modal + 1e-6

    return cossim_modal

def calc_loss_prior(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum()+1e-6).log_()))
    loss = per_embedding_loss.sum()    
    return loss, per_embedding_loss

def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    pos = (torch.exp(pos).sum(dim=1) + 1e-6).log_()
    # add the neg to the output 
    # neg = (torch.exp(neg).sum() + 1e-6).log_()
    return loss, per_embedding_loss, -pos.sum(), neg.sum()


def loss_ft(f_tensor, device, is_cossim=False):
    # to calculate frequency feature loss across frames
    # the input tensor is (batch size, num_feature_frames, length_of_feature)
    # num_feature_frames = t_len // n_feature_window
    num_feature_frames = f_tensor.shape[1]
    feature_sums = torch.sum(f_tensor, dim=1, keepdim=True)
    f_means_ex_per_feature = (feature_sums - f_tensor) / (num_feature_frames - 1)
    if is_cossim:
        # calculate cosine similarity
        f_dis_diff_per_feature = F.cosine_similarity(f_means_ex_per_feature, f_tensor, dim=2)
        f_means = torch.sum(torch.mean(f_dis_diff_per_feature, dim=1))
    else:
        # calculate Euc distance
        f_dis_diff_per_feature = f_means_ex_per_feature - f_tensor
        f_dis_diff_sq_sum_per_feature = torch.sum(torch.square(f_dis_diff_per_feature), dim=2)
        f_dis = torch.sqrt(f_dis_diff_sq_sum_per_feature)
        f_means = torch.sum(torch.mean(torch.sqrt(f_dis),dim=1))
    return f_means.to(device)




def cal_contrast_loss(sim_matrix, device):
    same_idx = list(range(sim_matrix.size(0)))
    sig_sim_matrix = 1 / (1 + torch.exp(-sim_matrix))
    pos = sig_sim_matrix[same_idx, :, same_idx]
    neg = torch.zeros((sim_matrix.size(0), sim_matrix.size(1))).to(device=device)
    for i in range(sim_matrix.size(0)):
        inf_matrix = torch.full((sim_matrix.size(1), sim_matrix.size(0)), float('-inf'))
        temp_tensor = sig_sim_matrix[i, :, :].clone()
        temp_tensor[:, i] = inf_matrix[:, i]
        max_vals, _ = torch.max(temp_tensor, dim=-1)
        neg[i, :] = max_vals

    per_embedding_loss = torch.ones((sim_matrix.size(0), sim_matrix.size(1))).to(device=device) - pos + neg
    loss = per_embedding_loss.sum()

    return loss, per_embedding_loss

def cal_intra_loss(sim_matrix, device):
    # calculate the intra loss between the centroid of modality A and the embedding vectors of modality B
    # only applicable when the sim matrix is calculated based on different modalities
    n_users = sim_matrix.size(0)
    n_utterance = sim_matrix.size(1)
    same_idx = list(range(n_users))
    per_user_sim = sim_matrix[same_idx, :, same_idx] 
    per_user_loss = torch.ones((n_users, n_utterance)).to(device=device) - torch.sigmoid(per_user_sim)
    loss = per_user_loss.sum()

    return loss, per_user_loss


def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized

# def mfccs_and_spec(wav_file, wav_process = False, calc_mfccs=False, calc_mag_db=False):    
#     sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
#     window_length = int(hp.data.window*hp.data.sr)
#     hop_length = int(hp.data.hop*hp.data.sr)
#     duration = hp.data.tisv_frame * hp.data.hop + hp.data.window
    
#     # Cut silence and fix length
#     if wav_process == True:
#         sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
#         length = int(hp.data.sr * duration)
#         sound_file = librosa.util.fix_length(sound_file, length)
        
#     spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
#     mag_spec = np.abs(spec)
    
#     mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
#     mel_spec = np.dot(mel_basis, mag_spec)
    
#     mag_db = librosa.amplitude_to_db(mag_spec)
#     #db mel spectrogram
#     mel_db = librosa.amplitude_to_db(mel_spec).T
    
#     mfccs = None
#     if calc_mfccs:
#         mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T
    
#     return mfccs, mel_db, mag_db

def pick_n_utterances(time_stamp, data_file_pth, train_ratio, user_n):
    # training data - user number - ratio in percentage - timestamp
    data_folder_pth = "./training_data/"
    os.makedirs(data_folder_pth, exist_ok=True)
    data_folder_pth = data_folder_pth + str(user_n) + "/"
    os.makedirs(data_folder_pth, exist_ok=True)
    data_folder_pth = data_folder_pth + str(int(train_ratio*100)) + "/"
    os.makedirs(data_folder_pth, exist_ok=True)
    data_folder_pth = data_folder_pth + time_stamp + "/"
    os.makedirs(data_folder_pth, exist_ok=True)
    # load the user audio/piezo files
    final_train_data = None
    final_test_data = None
    train_utter_n_lst = list()
    test_utter_n_lst = list()
    for i in range(user_n):
        piezo_pth = data_file_pth + "{}_piezo.npy".format(i)
        audio_pth = data_file_pth + "{}_audio.npy".format(i)

        p_data = np.load(piezo_pth)
        a_data = np.load(audio_pth)

        # concatenate p & a data into the same dim so that we could extract them simultaneously
        p_n_a_data = np.concatenate((p_data, a_data), axis=1)

        # randomly pick up utter_n records
        utter_n = p_n_a_data.shape[0] # number of utterance for user i
        training_utter_n = int(utter_n * train_ratio)
        train_utter_n_lst += [i for _ in range(training_utter_n)]
        test_utter_n_lst += [i for _ in range(utter_n - training_utter_n)]
        train_utter_idx = random.sample(range(p_n_a_data.shape[0]), training_utter_n)
        test_utter_idx = list(set(list(range(p_n_a_data.shape[0]))) - set(train_utter_idx))

        train_pa_data = p_n_a_data[train_utter_idx]
        test_pa_data = p_n_a_data[test_utter_idx]

        if i == 0:
            final_train_data = train_pa_data
            final_test_data = test_pa_data
        else:
            final_train_data = np.concatenate((final_train_data, train_pa_data), axis=0)
            final_test_data = np.concatenate((final_test_data, test_pa_data), axis=0)
    
    # save the data into the files
    np.save(data_folder_pth+'train_data.npy', final_train_data)
    np.save(data_folder_pth+'test_data.npy', final_test_data)
    np.save(data_folder_pth+'train_n_lst.npy', np.array(train_utter_n_lst))
    np.save(data_folder_pth+'test_n_lst.npy', np.array(test_utter_n_lst))

def binary_search(array, value):
    left, right = 0, len(array) - 1
    result = 0  # Default value if no suitable element is found
    
    while left <= right:
        mid = left + (right - left) // 2
        if array[mid] == value:
            return mid
        elif array[mid] < value:
            result = mid  # Update the result when found a smaller value
            left = mid + 1
        else:
            right = mid - 1
    
    return result


def normalize_tensors_based_audio(tensor_p, tensor_a):
    '''
    params: tensor p, tensor a
    return: normalized tensor p, tensor a
    '''
    b, f, t = tensor_a.shape
    tensor_a.contiguous()
    tensor_p.contiguous()
    max_a, _ = torch.max(tensor_a.reshape((b, -1)), dim=1, keepdim=True)
    min_a, _ = torch.min(tensor_a.reshape((b, -1)), dim=1, keepdim=True)

    # add eps 
    eps = 1e-10
    norm_denorm = max_a - min_a + eps

    # normalize the vectors
    tensor_p = (tensor_p - min_a.reshape((b, 1, 1))) / norm_denorm.reshape((b, 1, 1))
    tensor_a = (tensor_a - min_a.reshape((b, 1, 1))) / norm_denorm.reshape((b, 1, 1))
    return tensor_p, tensor_a


def pairwise_cos_sim(tensor_a, tensor_b):
    tensor_a.contiguous()
    b, u, _ = tensor_a.shape
    tensor_b.contiguous()

    tensor_a = tensor_a.repeat((1, b*u, 1))
    tensor_b = tensor_b.repeat((b*u, 1, 1))

    tensor_a = tensor_a.view(b * b * u * u, -1)
    tensor_b = tensor_b.view(b * b * u * u, -1)
    cos_sim = F.cosine_similarity(tensor_a, tensor_b)
    cos_sim.contiguous()
    cos_sim = cos_sim.view(b*u, b*u)
    return cos_sim


def random_split_tensor(input_tensor, split_n, device):
    '''
    Create the sublists of the given tensor with split_n
    '''
    N, U, E = input_tensor.shape
    if U < split_n:
        raise ValueError("Not enough utterances to be splited!")


    # randomly shuffle the utterance order list
    indices = torch.randperm(U).to(device)

    # get the number of sublists
    if U % split_n != 0:
        raise ValueError("The utterances cannot be evenly splitted.")
    
    sublist_size = U // split_n

    shuffled_tensor = torch.gather(input_tensor, 1, indices.expand_as(input_tensor))
    
    return shuffled_tensor, sublist_size

def softmax_loss(input_tensor, device):
    '''
    Calculate the SoftMax loss of the input tensor
    loss = -Sii + log\sum(\exp(Sij))
    '''
    N, M = input_tensor.shape
    if N != M:
        raise ValueError("The input tensor doesn't have identical length on different dims.")
    pos = torch.diag(input_tensor)

    # create a mask with ones everywhere except the diagonal elements
    mask = torch.ones_like(input_tensor).to(device) - torch.eye(N, requires_grad=True).to(device)
    mask.to(device)

    masked_tensor = input_tensor * mask.to(device)
    neg = (torch.exp(masked_tensor).sum(dim=1) + 1e-6).log_()
    loss_per_user_utter = -1 * (pos - neg)
    loss = loss_per_user_utter.mean()

    return loss, loss_per_user_utter

def cal_EER_coverter(sim_matrix):
    '''
    Calculate the EER, FAR, FRR of the input tensor
    '''
    N, M = sim_matrix.shape
    if N != M:
        raise ValueError("The input tensor doesn't have identical length on different dims.")
    
    # Initialize values
    diff = float('inf')
    EER = 0.0
    threshold = None
    EER_FAR = 0.0
    EER_FRR = 0.0

    # Iterate over potential thresholds
    for thres in torch.linspace(0.01, 1.0, 101):
        sim_matrix_thresh = sim_matrix > thres

        # Compute FAR and FRR
        FRR = 1 - torch.diag(sim_matrix_thresh).sum() / N
        
        FAR = (sim_matrix_thresh.sum() - torch.diag(sim_matrix_thresh).sum()) / N / (N - 1)

        # Update if this is the closest FAR and FRR we've seen so far
        if diff > abs(FAR - FRR):
            diff = abs(FAR - FRR)
            EER = ((FAR + FRR) / 2).item()
            threshold = thres.item()
            EER_FAR = FAR.item()
            EER_FRR = FRR.item()

    return EER, threshold, EER_FAR, EER_FRR

if __name__ == "__main__":
    tensor_a = torch.randn(10, 20, 192)
    tensor_b = torch.randn(10, 20, 192)
    # c = get_cossim_across_same_user(tensor_a, tensor_b)
    # print(c.shape)
    cos = pairwise_cos_sim(tensor_a, tensor_b)
    print(cos.shape)
    loss, loss_per_user_utter = softmax_loss(cos, torch.device('cpu'))
    print(loss)
    print(loss_per_user_utter.shape)
    cal_EER_coverter(cos)
    pass
