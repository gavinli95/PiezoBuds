import os
import sys
import time
import wandb
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import *
from my_models import *
from UniqueDraw import UniqueDraw
from utils import *
import torchvision

def compute_EER(sim_matrix):
    """
    Compute EER, FAR, FRR and the threshold at which EER occurs.

    Args:
    - sim_matrix (torch.Tensor): A similarity matrix of shape 
      (num of speakers, num of utterances, num of speakers).

    Returns:
    - EER (float): Equal error rate.
    - threshold (float): The threshold at which EER occurs.
    - FAR (float): False acceptance rate at EER.
    - FRR (float): False rejection rate at EER.
    """
    num_of_speakers, num_of_utters, _ = sim_matrix.shape
    
    # Initialize values
    diff = float('inf')
    EER = 0.0
    threshold = 0.0
    EER_FAR = 0.0
    EER_FRR = 0.0

    # Iterate over potential thresholds
    for thres in torch.linspace(0.5, 1.0, 50):
        sim_matrix_thresh = sim_matrix > thres

        # Compute FAR and FRR
        FAR = sum([(sim_matrix_thresh[i].sum() - sim_matrix_thresh[i, :, i].sum()).float()
                    for i in range(num_of_speakers)]) / (num_of_speakers - 1.0) / (num_of_utters) / num_of_speakers

        FRR = sum([(num_of_utters - sim_matrix_thresh[i, :, i].sum()).float()
                   for i in range(num_of_speakers)]) / (num_of_utters) / num_of_speakers

        # Update if this is the closest FAR and FRR we've seen so far
        if diff > abs(FAR - FRR):
            diff = abs(FAR - FRR)
            EER = ((FAR + FRR) / 2).item()
            threshold = thres.item()
            EER_FAR = FAR.item()
            EER_FRR = FRR.item()

    return EER, threshold, EER_FAR, EER_FRR

def train_and_test_model(device, model, ge2e_loss, train_set, test_set, optimizer, a_res, num_epochs=2000):
    # number of user for train and test in each epoch. Train 4 users, test 2 users.
    train_batch_size = 10
    test_batch_size = 10
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # train and test model
        # for phase in ['train', 'test']:
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                ge2e_loss.train()
                model.train()
                dataloader = train_loader
            else:
                # set model to test
                ge2e_loss.eval()
                model.eval()
                dataloader = test_loader
            
            # train each batch
            num_of_batches = 0
            loss_avg_batch_all = 0.0
            loss_avg_batch_aa = 0.0
            loss_avg_batch_pp = 0.0

            EERs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
            EER_FARs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
            EER_FRRs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
            EER_threshes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)

            for batch_id, (utterance, ids) in enumerate(dataloader):
                # get shape of input
                # number of speakers, number of utterances
                # features * 2 (256 * 2), time frame (1000)
                num_of_speakers, num_of_utters, w, l = utterance.shape
                num_of_batches = num_of_batches + 1

                with torch.set_grad_enabled(phase == 'train') and torch.autograd.set_detect_anomaly(True):
                    
                    if phase == 'train':
                        # training
                        # batch_size, time frame, length of feature
                        utterance = torch.reshape(utterance, (num_of_speakers * num_of_utters, w, l)).to(device)
                        audios = utterance[: , : a_res, : ].to(device)
                        audios = audios.permute(0, 2, 1).to(device)
                        audios = audios.unsqueeze(1)
                        embeddings_audio = model(audios)

                        # reshape the output to (number of speaker, number of utter, length of feature)
                        embeddings_audio = embeddings_audio.contiguous()
                        embeddings_audio = embeddings_audio.reshape([num_of_speakers, num_of_utters, -1])
                        # embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters, -1))
                        # embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters, -1))

                        # loss calculation
                        # loss_aa, _ = ge2e_loss(embeddings_audio, embeddings_audio)
                        loss_aa = ge2e_loss(embeddings_audio)
                        loss_extractor = loss_aa
                        # loss_extractor = loss_aa + loss_pp
                        # loss_extractor = loss_aa + loss_pp + loss_combined
                        loss_avg_batch_all += loss_extractor.item()
                        optimizer.zero_grad()
                        loss_extractor.backward()
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
                        optimizer.step()

                    if phase == 'test':
                        num_of_speakers, num_of_utters, w, l = utterance.shape
                        enrollment_batch, verification_batch = torch.split(utterance, int(num_of_utters/2), dim=1)

                        enrollment_batch = torch.reshape(enrollment_batch, (num_of_speakers * num_of_utters // 2, w, l)).to(device)
                        enrollment_batch = enrollment_batch.permute(0, 2, 1)
                        enrollment_audio = enrollment_batch[: , : , : a_res].to(device)
                        verification_batch = torch.reshape(verification_batch, (num_of_speakers * num_of_utters // 2, w, l)).to(device)
                        verification_batch = verification_batch.permute(0, 2, 1)
                        verification_audio = verification_batch[: , : , : a_res].to(device)
                        

                        enrollment_audio = enrollment_audio.unsqueeze(1)
                        embeddings_audio = model(enrollment_audio)
                        
                        embeddings_audio = embeddings_audio.contiguous()
                        embeddings_audio = embeddings_audio.reshape([num_of_speakers, num_of_utters, -1])

                        centroids_audio = get_centroids(embeddings_audio)

                        verification_audio = verification_audio.unsqueeze(1)
                        embeddings_audio = model(verification_audio)
                        
                        embeddings_audio = embeddings_audio.contiguous()
                        embeddings_audio = embeddings_audio.reshape([num_of_speakers, num_of_utters, -1])

                        # loss_aa, _ = ge2e_loss(embeddings_audio, embeddings_audio)
                        loss_aa = ge2e_loss(embeddings_audio)
                        loss_extractor = loss_aa
                        loss_avg_batch_all += loss_extractor.item()

                        # EER of Audio
                        sim_matrix = get_cossim(embeddings_audio, centroids_audio)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[1] += EER
                        EER_FARs[1] += EER_FAR
                        EER_FRRs[1] += EER_FRR
                        EER_threshes[1] += EER_thresh

            epoch_loss_all = loss_avg_batch_all / num_of_batches

            print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')

            wandb.log({'epoch': epoch, f'Loss/{phase}_extractor': epoch_loss_all})

            if phase == 'test':
                EERs /= float(num_of_batches)
                EER_FARs /= float(num_of_batches)
                EER_FRRs /= float(num_of_batches)
                EER_threshes /= float(num_of_batches)

                print("\nCentroids: Audio  Verification Input: Audio "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[1], EER_threshes[1], EER_FARs[1], EER_FRRs[1]))
                wandb.log({'epoch': epoch, 'EER/C_A_VI_A': EERs[1], 'FAR/C_A_VI_A': EER_FARs[1], 'FRR/C_A_VI_A': EER_FRRs[1]})
                wandb.log({'epoch': epoch, 'threshold/C_A_VI_A': EER_threshes[1]})

    return model 


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # get the training and test data set
    ori_ids = list(range(1122))
    ids = random.sample(ori_ids, 100)
    valid_ids = random.sample(ids, 20)
    # ids = list(range(22))
    # valid_ids = [3, 6, 8, 1]
    # valid_ids = [11, 12, 14, 16]
    train_ids = [item for item in ids if item not in valid_ids]
    print(train_ids)
    print(valid_ids)
    
    data_file_dir = './processed_data/voxceleb/res_256/' # folder where stores the data for training and test
    pth_store_dir = './pth_model/'
    os.makedirs(pth_store_dir, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------
    
    lr = 0.001
    num_of_epoches = 2000

    # comment = 'tf_vox_old_data_model' # simple descriptions of specifications of this model, for example, 't_f' means we use the model which contains time and frequency nn layers
    comment = 'tf_vox_data_model_mobilenetv3'

    ge2e_loss = GE2ELoss_ori(device)
    a_res = 256
    # extractor = Extractor_w_F_1D(in_channel_n=128, device=device, f_len=a_res).to(device)
    # extractor = Extractor_w_TF(device=device, t_len=128, f_len=a_res).to(device)
    # extractor = extractor1DCNN(device=device, in_feature=256, out_feature=64, channel_number=128).to(device)
    extractor = torchvision.models.mobilenet_v3_small(pretrained=True)
    extractor.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    extractor.classifier[3] = nn.Linear(extractor.classifier[3].in_features, 128)
    extractor.to(device)
    optimizer = torch.optim.SGD([
        {'params': extractor.parameters()},
        {'params': ge2e_loss.parameters()}
    ], lr=lr)
    model_n = 1
    
    # create the folder to store the model
    model_struct = 'model_n_' + str(model_n) + '_res_a' + str(a_res) + '_' + comment
    # initialize the wandb configuration
    time_stamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    wandb.init(
        # team name
        entity="piezobuds",
        # set the project name
        project="PiezoBuds",
        # params of the task
        name=model_struct+'_'+time_stamp
    )
    model_store_pth = pth_store_dir + model_struct + '/'
    os.makedirs(model_store_pth, exist_ok=True)
    model_final_path = model_store_pth + time_stamp + '/'
    os.makedirs(model_final_path, exist_ok=True)
    
    # load the data 
    train_set = VoxSTFTFeatureSet(False, train_ids, 20, data_file_dir)
    test_set = VoxSTFTFeatureSet(False, valid_ids, 20, data_file_dir)
    
    
    extractor = train_and_test_model(device=device, model=extractor, ge2e_loss=ge2e_loss, 
                                 train_set=train_set, test_set=test_set,
                                 optimizer=optimizer, a_res=a_res, num_epochs=num_of_epoches)
    
    torch.save(extractor.state_dict(), model_final_path+'m.pth')