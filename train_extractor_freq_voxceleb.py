import os
import sys
import time
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import *
from my_models import *
from torch.utils.tensorboard import SummaryWriter
from UniqueDraw import UniqueDraw
from utils import *
import torchvision
from mobile_net_v3 import *


def hook_fn(module, input, output):
    """ Store the output of the hook """
    global hooked_output
    hooked_output = output


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
    for thres in torch.linspace(0.5, 1.0, 501):
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

def train_and_test_model(device, model, class_model, loss_func, data_set, optimizer, train_ratio=0.8, num_epochs=2000, num_feature_window=16, num_cls_window=2):
    # writer = SummaryWriter()
    # number of user for train and test in each epoch. Train 4 users, test 2 users.
    train_batch_size = 128
    test_batch_size = 128
    data_size = len(data_set)
    train_size = int(data_size * train_ratio)
    test_size = data_size - train_size
    train_tmp_set, test_tmp_set = torch.utils.data.random_split(data_set, [train_size, test_size])
    train_loader = DataLoader(train_tmp_set, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_tmp_set, batch_size=test_batch_size, shuffle=True, drop_last=True)


    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # train and test model
        # for phase in ['train', 'test']:
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                model.train()
                class_model.train()
                dataloader = train_loader
            else:
                # set model to test
                model.eval()
                class_model.eval()
                dataloader = test_loader
            
            # train each batch
            num_of_batches = 0
            loss_avg_batch_all = 0.0
            acc = 0.0

            for batch_id, (utterance, ids) in enumerate(dataloader):
                # get shape of input
                batch_size, f_len, t_len = utterance.shape
                # num_of_speakers, num_of_utters, w, l = utterance.shape
                # num_of_batches = num_of_batches + 1

                with torch.set_grad_enabled(phase == 'train') and torch.autograd.set_detect_anomaly(True):
                    
                    if phase == 'train':
                        # training
                        # batch_size, time frame, length of feature
                        audios = utterance.permute(0, 2, 1).to(device)
                        n_feature_frames = t_len // num_feature_window
                        audios = audios.contiguous()
                        audios = audios.view((batch_size * n_feature_frames, num_feature_window, f_len))
                        audios = audios.unsqueeze(1)
                        
                        _ = extractor(audios)
                        output_from_features_minus_1 = hooked_output
                        embeddings_audios = torch.clone(torch.flatten(output_from_features_minus_1, 1)) # (batch_size * n_feature_frames, F)
                        embeddings_audios.contiguous()
                        embeddings_audios = embeddings_audios.view((batch_size, n_feature_frames, -1))

                        # loss freq
                        loss_feq_t = loss_ft(embeddings_audios, device, is_cossim=True)

                        # loss class
                        embeddings_audios = embeddings_audios.view((batch_size, n_feature_frames // num_cls_window, num_cls_window, -1))
                        embeddings_audios = torch.mean(embeddings_audios, dim=2) # (batch_size, n_feature_frames // num_cls_window, 1, F)
                        embeddings_audios = embeddings_audios.view((batch_size * n_feature_frames // num_cls_window, -1)) # (batch_size, n_feature_frames // num_cls_window, F)
                        pred_ids = class_model(embeddings_audios)
                        ids = ids.contiguous()
                        ids_expand = ids.view((-1, 1))
                        ids_expand = ids_expand.repeat(1, n_feature_frames // num_cls_window)
                        ids_gpu = ids_expand.flatten()
                        ids_gpu = ids_gpu.to(device)
                        loss_class = loss_func(pred_ids, ids_gpu)

                        acc += np.sum(np.argmax(pred_ids.cpu().data.numpy(), axis=1) == ids_gpu.cpu().data.numpy())

                        # loss_extractor = loss_class + loss_feq_t
                        loss_extractor = loss_class

                        loss_avg_batch_all += loss_extractor.item()
                        optimizer.zero_grad()
                        loss_extractor.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                        optimizer.step()

                    if phase == 'test':
                        audios = utterance.permute(0, 2, 1).to(device)
                        n_feature_frames = t_len // num_feature_window
                        audios = audios.contiguous()
                        audios = audios.view((batch_size * n_feature_frames, num_feature_window, f_len))
                        audios = audios.unsqueeze(1)
                        
                        _ = extractor(audios)
                        output_from_features_minus_1 = hooked_output
                        embeddings_audios = torch.clone(torch.flatten(output_from_features_minus_1, 1)) # (batch_size * n_feature_frames, F)
                        embeddings_audios.contiguous()
                        embeddings_audios = embeddings_audios.view((batch_size, n_feature_frames, -1))

                        # loss freq
                        loss_feq_t = loss_ft(embeddings_audios, device, is_cossim=True)

                        # loss class
                        embeddings_audios = embeddings_audios.view((batch_size, n_feature_frames // num_cls_window, num_cls_window, -1))
                        embeddings_audios = torch.mean(embeddings_audios, dim=2) # (batch_size, n_feature_frames // num_cls_window, 1, F)
                        embeddings_audios = embeddings_audios.view((batch_size * n_feature_frames // num_cls_window, -1)) # (batch_size, n_feature_frames // num_cls_window, F)
                        pred_ids = class_model(embeddings_audios)
                        ids = ids.contiguous()
                        ids_expand = ids.view((-1, 1))
                        ids_expand = ids_expand.repeat(1, n_feature_frames // num_cls_window)
                        ids_gpu = ids_expand.flatten()
                        ids_gpu = ids_gpu.to(device)
                        loss_class = loss_func(pred_ids, ids_gpu)

                        acc += np.sum(np.argmax(pred_ids.cpu().data.numpy(), axis=1) == ids_gpu.cpu().data.numpy())

                        # loss_extractor = loss_class + loss_feq_t
                        loss_extractor = loss_class
                        
                        loss_avg_batch_all += loss_extractor.item()

            epoch_loss_all = loss_avg_batch_all / len(dataloader)

            if phase == 'train':
                epoch_acc = acc / (len(dataloader) * train_batch_size * n_feature_frames // num_cls_window)
                print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')
                print(f'{phase} Acc: {epoch_acc:.4f}')
                wandb.log({'epoch': epoch, f'Loss/{phase}_extractor': epoch_loss_all})
                wandb.log({'epoch': epoch, f'Acc/{phase}': epoch_acc}) 
                
            if phase == 'test':
                epoch_acc = acc / (len(dataloader) * test_batch_size * n_feature_frames // num_cls_window)
                print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')
                print(f'{phase} Acc: {epoch_acc:.4f}')
                wandb.log({'epoch': epoch, f'Loss/{phase}_extractor': epoch_loss_all})
                wandb.log({'epoch': epoch, f'Acc/{phase}': epoch_acc}) 

    return (model, class_model)


if __name__ == "__main__":

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    data_file_dir = './processed_data/voxceleb/res_256/' # folder where stores the data for training and test
    pth_store_dir = './pth_model/'
    os.makedirs(pth_store_dir, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------
    
    lr = 0.001
    n_user = 100
    train_ratio = 0.8
    num_of_epoches = 2000
    num_feature_windows = 32 # extract feature using num_feature_windows STFT frame
    num_class_windows = 1 # conduct classification using num_class_windows feature frame (mean across feature frames)

    comment = 'mobilenetv3small_hop_256_t_16_class_stft_dis_cos' # simple descriptions of specifications of this model, for example, 't_f' means we use the model which contains time and frequency nn layers

    # extractor = extractor1DCNN(device=device, channel_number=160, in_feature=256, out_feature=128).to(device)

    extractor = torchvision.models.mobilenet_v3_large(pretrained=False)
    extractor.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    extractor.classifier[3] = nn.Linear(extractor.classifier[3].in_features, 256)

    # extractor = mobilenet_v3_large(reduced_tail=False)

    hook = extractor.avgpool.register_forward_hook(hook_fn)

    # extractor = SpeechEmbedder().to(device)
    extractor.to(device)
    class_model = class_model_FC(device=device, input_len=960, class_n=n_user).to(device)
    # optimizer = torch.optim.SGD([
    #     {'params': extractor.parameters()},
    #     {'params': class_model.parameters()}
    # ], lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    optimizer = torch.optim.Adam([
        {'params': extractor.parameters()},
        {'params': class_model.parameters()}
    ], lr=lr)
    
    # create the folder to store the model
    model_struct = 'model_nfw{}_ncw{}_user{}_'.format(num_feature_windows, num_class_windows, n_user) + comment
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
    
    # create the data set for training and test
    # pick_n_utterances(time_stamp=time_stamp, data_file_pth=data_file_dir, train_ratio=train_ratio, user_n=n_user)

    # load the data 
    voxceleb_set = VoxcelebSet4Class(n_user=n_user, dir_root=data_file_dir, is_audio=True)
    # test_set = STFTFeatureSet4Class(n_user=n_user, train_ratio=train_ratio, time_stamp=time_stamp, train=False)
    print(len(voxceleb_set))
    # print(len(test_set))

    loss_func = nn.CrossEntropyLoss()
    
    
    (extractor, class_model) = train_and_test_model(device=device, model=extractor, class_model=class_model,
                                                    loss_func=loss_func, data_set=voxceleb_set, optimizer=optimizer,
                                                    train_ratio=train_ratio, num_epochs=num_of_epoches,
                                                    num_feature_window=num_feature_windows, num_cls_window=num_class_windows)
    
    
    torch.save(extractor.state_dict(), model_final_path+'m.pth')
    torch.save(class_model.state_dict(), model_final_path+'c.pth')
