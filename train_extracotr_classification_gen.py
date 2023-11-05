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

def train_and_test_model(device, model, class_model, loss_func, train_set, test_set, optimizer, a_res, p_res, num_epochs=2000, train_seperate=False):
    # writer = SummaryWriter()
    # number of user for train and test in each epoch. Train 4 users, test 2 users.
    train_batch_size = 128
    test_batch_size = 128
    data_size = len(train_set)
    train_size = int(data_size * 0.9)
    test_size = data_size - train_size
    train_tmp_set, test_tmp_set = torch.utils.data.random_split(train_set, [train_size, test_size])
    train_loader = DataLoader(train_tmp_set, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_tmp_set, batch_size=test_batch_size, shuffle=True, drop_last=True)


    if train_seperate:
        model_a, model_p = model

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # train and test model
        # for phase in ['train', 'test']:
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                if train_seperate == False:
                    model.train()
                else:
                    model_a.train()
                    model_p.train()
                class_model.train()
                dataloader = train_loader
            else:
                # set model to test
                if train_seperate == False:
                    model.eval()
                else:
                    model_a.eval()
                    model_p.eval()
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
                        # utterance = torch.reshape(utterance, (num_of_speakers * num_of_utters, w, l)).to(device)
                        piezos = utterance[: , : p_res, : ].to(device)
                        audios = utterance[: , f_len // 2: f_len // 2 + a_res, : ].to(device)
                        piezos = piezos.permute(0, 2, 1).to(device)
                        piezos = piezos.unsqueeze(1)
                        audios = audios.permute(0, 2, 1).to(device)
                        audios = audios.unsqueeze(1)

                        combined = torch.cat((audios, piezos), dim=1)
                        embeddings_combined = extractor(combined)
                        output_from_features_minus_1 = hooked_output
                        embeddings_combined = torch.clone(torch.flatten(output_from_features_minus_1, 1))

                        # if train_seperate == False:
                        #     embeddings_piezo = model(piezos)
                        #     embeddings_audio = model(audios)
                        # else:
                        #     embeddings_piezo = model_p(piezos)
                        #     embeddings_audio = model_a(audios)

                        # loss calculation
                        # embeddings_combined = torch.cat((embeddings_piezo, embeddings_audio), dim=1)
                        # pred_ids = class_model(embeddings_audio, embeddings_piezo)
                        pred_ids = class_model(embeddings_combined)
                        ids_gpu = ids.to(device)
                        loss_class = loss_func(pred_ids, ids_gpu)
                        acc += np.sum(np.argmax(pred_ids.cpu().data.numpy(), axis=1) == ids_gpu.cpu().data.numpy())
                        loss_extractor = loss_class
                        loss_avg_batch_all += loss_extractor.item()
                        optimizer.zero_grad()
                        loss_extractor.backward()

                        if train_seperate == False:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                            # torch.nn.utils.clip_grad_norm_(class_model.parameters(), 10.0)
                        else:
                            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 3.0)
                            torch.nn.utils.clip_grad_norm_(model_p.parameters(), 3.0)

                        optimizer.step()

                    if phase == 'test':
                        piezos = utterance[: , : p_res, : ].to(device)
                        audios = utterance[: , f_len // 2: f_len // 2 + a_res, : ].to(device)
                        piezos = piezos.permute(0, 2, 1).to(device)
                        piezos = piezos.unsqueeze(1)
                        audios = audios.permute(0, 2, 1).to(device)
                        audios = audios.unsqueeze(1)
                        combined = torch.cat((audios, piezos), dim=1)
                        embeddings_combined = extractor(combined)

                        output_from_features_minus_1 = hooked_output
                        embeddings_combined = torch.clone(torch.flatten(output_from_features_minus_1, 1))

                        # if train_seperate == False:
                        #     embeddings_piezo = model(piezos)
                        #     embeddings_audio = model(audios)
                        # else:
                        #     embeddings_piezo = model_p(piezos)
                        #     embeddings_audio = model_a(audios)

                        # loss calculation
                        # embeddings_combined = torch.cat((embeddings_piezo, embeddings_audio), dim=1)
                        # pred_ids = class_model(embeddings_audio, embeddings_piezo)
                        pred_ids = class_model(embeddings_combined)
                        ids_gpu = ids.to(device)
                        loss_class = loss_func(pred_ids, ids_gpu)
                        acc += np.sum(np.argmax(pred_ids.cpu().data.numpy(), axis=1) == ids_gpu.cpu().data.numpy())
                        # print(np.argmax(pred_ids.cpu().data.numpy(), axis=1))
                        # print(ids_gpu.cpu().data.numpy())
                        loss_extractor = loss_class
                        loss_avg_batch_all += loss_extractor.item()
                        

            epoch_loss_all = loss_avg_batch_all / len(dataloader)

            if phase == 'train':
                epoch_acc = acc / (len(dataloader) * train_batch_size)
                print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')
                print(f'{phase} Acc: {epoch_acc:.4f}')
                wandb.log({'epoch': epoch, f'Loss/{phase}_extractor': epoch_loss_all})
                wandb.log({'epoch': epoch, f'Acc/{phase}': epoch_acc}) 
                
            if phase == 'test':
                epoch_acc = acc / (len(dataloader) * test_batch_size)
                print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')
                print(f'{phase} Acc: {epoch_acc:.4f}')
                wandb.log({'epoch': epoch, f'Loss/{phase}_extractor': epoch_loss_all})
                wandb.log({'epoch': epoch, f'Acc/{phase}': epoch_acc}) 

            
            

    return (model, class_model) if not train_seperate else (model_a, model_p, class_model)


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    data_file_dir = './processed_data/power_spectra/res_256_hop_256_t_16/' # folder where stores the data for training and test
    pth_store_dir = './pth_model/'
    os.makedirs(pth_store_dir, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------
    
    lr = 0.001
    n_user = 49
    train_ratio = 1
    num_of_epoches = 2000
    train_seperate = False

    comment = 'mobilenetv3large1d_960_hop_256_t_16_class_pwr_spec_49u' # simple descriptions of specifications of this model, for example, 't_f' means we use the model which contains time and frequency nn layers

    if train_seperate == False:
        a_res = 256
        p_res = 256
        # extractor = extractor1DCNN(device=device, channel_number=160, in_feature=256, out_feature=128).to(device)

        extractor = torchvision.models.mobilenet_v3_small(pretrained=True)
        extractor.features[0][0] = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        extractor.classifier[3] = nn.Linear(extractor.classifier[3].in_features, 256)

        # extractor = mobilenet_v3_large(reduced_tail=False)

        hook = extractor.avgpool.register_forward_hook(hook_fn)

        # extractor = SpeechEmbedder().to(device)
        extractor.to(device)
        class_model = class_model_FC(device=device, input_len=576, class_n=n_user).to(device)
        # optimizer = torch.optim.SGD([
        #     {'params': extractor.parameters()},
        #     {'params': class_model.parameters()}
        # ], lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        optimizer = torch.optim.Adam([
            {'params': extractor.parameters()},
            {'params': class_model.parameters()}
        ], lr=lr)
        model_n = 1
    else:
        a_res = 512
        p_res = 256
        extractor_a = Extractor_w_LSTM(device=device, layer_n=5, input_dim=a_res, output_dim=a_res).to(device)
        extractor_p = Extractor_w_LSTM(device=device, layer_n=5, input_dim=p_res, output_dim=p_res).to(device)
        class_model = class_model_FC(device=device, input_len=a_res, class_n=n_user)
        extractor = (extractor_a, extractor_p)
        optimizer = torch.optim.SGD([
            {'params': extractor_a.parameters()},
            {'params': extractor_p.parameters()},
            {'params': class_model.parameters()}
        ], lr=lr)
        model_n = 2 # number of models: wish to train one model jointly or 2 models seperately
    
    # create the folder to store the model
    model_struct = 'model_n_' + str(model_n) + '_res_a' + str(a_res) + '_p' + str(p_res) + '_' + comment
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
    pick_n_utterances(time_stamp=time_stamp, data_file_pth=data_file_dir, train_ratio=train_ratio, user_n=n_user)

    # load the data 
    train_set = STFTFeatureSet4Class(n_user=n_user, train_ratio=train_ratio, time_stamp=time_stamp, train=True)
    test_set = STFTFeatureSet4Class(n_user=n_user, train_ratio=train_ratio, time_stamp=time_stamp, train=False)
    print(len(train_set))
    print(len(test_set))

    loss_func = nn.CrossEntropyLoss()
    
    
    (extractor, class_model) = train_and_test_model(device=device, model=extractor, class_model=class_model,
                                 loss_func=loss_func, train_set=train_set, test_set=test_set, 
                                 optimizer=optimizer, a_res=a_res, p_res=p_res, 
                                 num_epochs=num_of_epoches, train_seperate=train_seperate)
    
    if train_seperate == False:
        torch.save(extractor.state_dict(), model_final_path+'m.pth')
        torch.save(class_model.state_dict(), model_final_path+'c.pth')
    else:
        torch.save(extractor[0].state_dict(), model_final_path+'a.pth')
        torch.save(extractor[1].state_dict(), model_final_path+'p.pth')
        torch.save(class_model.state_dict(), model_final_path+'c.pth')