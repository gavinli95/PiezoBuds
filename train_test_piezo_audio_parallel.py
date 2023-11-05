import sys
import time
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import STFTFeatureSet
from my_models import CNNModel, LSTMWithAttention, GE2ELoss, ConverterNetwork, ConverterNetwork1DCNN, DeepSet, ConverterNetwork2D21D, extractor1DCNN
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from UniqueDraw import UniqueDraw
from utils import calc_loss, cal_intra_loss
from utils import get_centroids, get_cossim, get_modal_cossim, cosine_similarity, get_distance_matrix, draw_distance_matrix, get_centroids_kmeans


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


def train_and_test_model(device, model_audio, model_piezo, converter, ge2e_loss, is_cnn, train_set, test_set, optimizer, optimizer_converter, num_epochs=2000):
    writer = SummaryWriter()
    # number of user for train and test in each epoch. Train 4 users, test 2 users.
    batch_size = 2
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True)

    for epoch in range(num_epochs * 2):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # train and test model
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                model_audio.train()
                model_piezo.train()
                converter.train()
                dataloader = train_loader
            else:
                # set model to test
                model_audio.eval()
                model_piezo.eval()
                converter.eval()
                dataloader = test_loader
            
            # train each batch
            num_of_batches = 0
            loss_avg_batch_extractor = 0.0
            loss_avg_batch_converter = 0.0

            EERs = [0, 0, 0, 0, 0, 0, 0]
            EER_FARs = [0, 0, 0, 0, 0, 0, 0]
            EER_FRRs = [0, 0, 0, 0, 0, 0, 0]
            EER_threshes = [0, 0, 0, 0, 0, 0, 0]

            for batch_id, (utterance, ids) in enumerate(dataloader):
                # get shape of input
                # number of speakers, number of utterances
                # features * 2 (256 * 2), time frame (1000)
                num_of_speakers, num_of_utters, w, l = utterance.shape
                num_of_batches = num_of_batches + 1

                with torch.set_grad_enabled(phase == 'train') and torch.autograd.set_detect_anomaly(True):
                    
                    if phase == 'train':
                        # training
                        if is_cnn:
                            # batch_size, 1, length of feature, time frame
                            utterance = torch.reshape(utterance, (num_of_speakers * num_of_utters, 1, w, l)).to(device)
                            piezos = utterance[:, :, :w//2, :].to(device)
                            audios = utterance[:, :, w//2:, :].to(device)
                        else:
                            # batch_size, time frame, length of feature
                            utterance = torch.reshape(utterance, (num_of_speakers * num_of_utters, w, l)).to(device)
                            piezos = utterance[:, :w//2, :].to(device)
                            audios = utterance[:, w//2:, :].to(device)
                            piezos = piezos.permute(0, 2, 1).to(device)
                            audios = audios.permute(0, 2, 1).to(device)
                        
                        embeddings_piezo, _ = model_piezo(piezos)
                        embeddings_audio, _ = model_audio(audios)
                        # embeddings_audio_for_converter = torch.clone(embeddings_audio)
                        # embeddings_audio_to_piezo = converter(embeddings_audio)
                        # embeddings_piezo_to_audio = converter(embeddings_piezo)
                        embeddings_x = converter(embeddings_audio, embeddings_piezo)
                        _, output_dim = embeddings_piezo.shape

                        # reshape the output to (number of speaker, number of utter, length of feature)
                        embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters, output_dim))
                        embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters, output_dim))
                        embeddings_x = torch.reshape(embeddings_x, (num_of_speakers, num_of_utters, output_dim))
                        # embeddings_audio_to_piezo = torch.reshape(embeddings_audio_to_piezo, (num_of_speakers, num_of_utters, output_dim))
                        # embeddings_piezo_to_audio = torch.reshape(embeddings_piezo_to_audio, (num_of_speakers, num_of_utters, output_dim))

                        centroids_x = get_centroids(embeddings_x)
                        # extractor
                        loss_aa, loss_pp = ge2e_loss(embeddings_piezo, embeddings_audio)
                        loss_x, loss_pp = ge2e_loss(embeddings_piezo, embeddings_x)
                        # loss_extractor = loss_aa + loss_pp + loss_x
                        loss_extractor = loss_aa + loss_pp
                        loss_avg_batch_extractor += loss_extractor.item()
                        # converter
                        sim_matrix_p2x = get_modal_cossim(embeddings_piezo, centroids_x)
                        sim_matrix_a2x = get_modal_cossim(embeddings_audio, centroids_x)
                        loss_converter_p2x, loss_converter_p2x_user = cal_intra_loss(sim_matrix_p2x, device)
                        loss_converter_a2x, loss_converter_a2x_user = cal_intra_loss(sim_matrix_a2x, device)
                        wp = 0.7
                        loss_converter = ((loss_converter_p2x_user * wp - loss_converter_a2x_user * (1 - wp)) ** 2).sum() + 0.5 * (wp * loss_converter_p2x + (1 - wp) * loss_converter_a2x) +  2 * loss_x
                        loss_avg_batch_converter += loss_converter.item()

                        loss = loss_extractor + 3 * loss_converter # not used when applying below codes

                        if int(epoch / 20) % 2 == 0:
                            optimizer.zero_grad()
                            # loss_extractor.backward()
                            loss_aa.backward()
                            loss_pp.backward()
                            torch.nn.utils.clip_grad_norm_(model_audio.parameters(), 3.0)
                            torch.nn.utils.clip_grad_norm_(model_piezo.parameters(), 3.0)
                            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
                            optimizer.step()
                        else:
                            optimizer_converter.zero_grad()
                            loss_converter.backward()
                            torch.nn.utils.clip_grad_norm_(converter.parameters(), 1.0)
                            optimizer_converter.step()

                    else:
                        # testing
                        # split data into enrollment and verification
                        enrollment_batch, verification_batch = torch.split(utterance, int(num_of_utters/2), dim=1)
                        if is_cnn:
                            # batch_size, 1, length of feature, time frame
                            enrollment_batch = torch.reshape(enrollment_batch, (num_of_speakers * num_of_utters // 2, 1, w, l)).to(device)
                            enrollment_piezos = enrollment_batch[:, :, :w//2, :].to(device)
                            enrollment_audios = enrollment_batch[:, :, w//2:, :].to(device)
                            verification_batch = torch.reshape(verification_batch, (num_of_speakers * num_of_utters // 2, 1, w, l)).to(device)
                            verification_piezos = verification_batch[:, :, :w//2, :].to(device)
                            verification_audios = verification_batch[:, :, w//2:, :].to(device)
                        else:
                            # batch_size, time frame, length of feature
                            enrollment_batch = torch.reshape(enrollment_batch, (num_of_speakers * num_of_utters // 2, w, l)).to(device)
                            enrollment_batch = enrollment_batch.permute(0, 2, 1)
                            enrollment_piezos = enrollment_batch[:, :, :w//2].to(device)
                            enrollment_audios = enrollment_batch[:, :, w//2:].to(device)
                            verification_batch = torch.reshape(verification_batch, (num_of_speakers * num_of_utters // 2, w, l)).to(device)
                            verification_batch = verification_batch.permute(0, 2, 1)
                            verification_piezos = verification_batch[:, :, :w//2].to(device)
                            verification_audios = verification_batch[:, :, w//2:].to(device)
                        
                        # get centroids from enrollment
                        embeddings_piezo, _ = model_piezo(enrollment_piezos)
                        embeddings_audio, _ = model_audio(enrollment_audios)
                        embeddings_x = converter(embeddings_piezo, embeddings_audio)
                        _, output_dim = embeddings_piezo.shape
                        embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters // 2, output_dim))
                        embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters // 2, output_dim))
                        embeddings_x = torch.reshape(embeddings_x, (num_of_speakers, num_of_utters // 2, output_dim))
                        centroids_piezo = get_centroids(embeddings_piezo)
                        centroids_audio = get_centroids(embeddings_audio)
                        centroids_x = get_centroids(embeddings_x)

                        # get verification embeddings
                        embeddings_piezo, _ = model_piezo(verification_piezos)
                        embeddings_audio, _ = model_audio(verification_audios)
                        embeddings_x = converter(embeddings_audio, embeddings_piezo)
                        embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters // 2, output_dim))
                        embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters // 2, output_dim))
                        embeddings_x = torch.reshape(embeddings_x, (num_of_speakers, num_of_utters // 2, output_dim))

                        # get EER using different combination
                        # C means centroids, VI means verification input

                        # C audio_to_piezo, VI piezp
                        sim_matrix = get_modal_cossim(embeddings_x, centroids_x)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[-1] += EER
                        EER_FARs[-1] += EER_FAR
                        EER_FRRs[-1] += EER_FRR
                        EER_threshes[-1] += EER_thresh

                        # C piezo, VI piezo
                        sim_matrix = get_modal_cossim(embeddings_piezo, centroids_piezo)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[0] += EER
                        EER_FARs[0] += EER_FAR
                        EER_FRRs[0] += EER_FRR
                        EER_threshes[0] += EER_thresh

                        # C audio, VI audio
                        sim_matrix = get_modal_cossim(embeddings_audio, centroids_audio)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[1] += EER
                        EER_FARs[1] += EER_FAR
                        EER_FRRs[1] += EER_FRR
                        EER_threshes[1] += EER_thresh

                        # C piezo, VI audio
                        sim_matrix = get_modal_cossim(embeddings_audio, centroids_piezo)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[2] += EER
                        EER_FARs[2] += EER_FAR
                        EER_FRRs[2] += EER_FRR
                        EER_threshes[2] += EER_thresh

                        # C audio, VI piezo
                        sim_matrix = get_modal_cossim(embeddings_piezo, centroids_audio)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[3] += EER
                        EER_FARs[3] += EER_FAR
                        EER_FRRs[3] += EER_FRR
                        EER_threshes[3] += EER_thresh

                        # C audio + piezo, VI audio + piezo
                        sim_matrix = get_modal_cossim(torch.cat((embeddings_audio, embeddings_piezo), dim=-1), 
                                                torch.cat((centroids_audio, centroids_piezo), dim=-1))
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[4] += EER
                        EER_FARs[4] += EER_FAR
                        EER_FRRs[4] += EER_FRR
                        EER_threshes[4] += EER_thresh

                        # C audio + piezo, VI piezo + audio
                        sim_matrix = get_modal_cossim(torch.cat((embeddings_piezo, embeddings_audio), dim=-1), 
                                                torch.cat((centroids_audio, centroids_piezo), dim=-1))
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[5] += EER
                        EER_FARs[5] += EER_FAR
                        EER_FRRs[5] += EER_FRR
                        EER_threshes[5] += EER_thresh
            wandb.define_metric("epoch")
            wandb.define_metric("Loss/*", step_metric="epoch")
            wandb.define_metric("EER/*", step_metric="epoch")
            wandb.define_metric("FAR/*", step_metric="epoch")
            wandb.define_metric("FRR/*", step_metric="epoch")
            if phase == 'train':
                epoch_loss_extractor = loss_avg_batch_extractor / num_of_batches
                print(f'{phase} Loss Extractor: {epoch_loss_extractor:.4f}')
                # writer.add_scalar('Loss/train', epoch_loss, epoch)
                wandb.log({'epoch': epoch, 'Loss/train_extractor': epoch_loss_extractor})

                epoch_loss_converter = loss_avg_batch_converter / num_of_batches
                print(f'{phase} Loss Converter: {epoch_loss_converter:.4f}')
                # writer.add_scalar('Loss/train', epoch_loss, epoch)
                wandb.log({'epoch': epoch, 'Loss/train_converter': epoch_loss_converter})
            else:
                EERs = np.array(EERs) / num_of_batches
                EER_FARs = np.array(EER_FARs) / num_of_batches
                EER_FRRs = np.array(EER_FRRs) / num_of_batches
                EER_threshes = np.array(EER_threshes) / num_of_batches

                print("\nCentroids: Piezo  Verification Input: Piezo "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[0], EER_threshes[0], EER_FARs[0], EER_FRRs[0]))
                wandb.log({'epoch': epoch, 'EER/C_P_VI_P': EERs[0], 'FAR/C_P_VI_P': EER_FARs[0], 'FRR/C_P_VI_P': EER_FRRs[0]})

                print("\nCentroids: Audio  Verification Input: Audio "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[1], EER_threshes[1], EER_FARs[1], EER_FRRs[1]))
                wandb.log({'epoch': epoch, 'EER/C_A_VI_A': EERs[1], 'FAR/C_A_VI_A': EER_FARs[1], 'FRR/C_A_VI_A': EER_FRRs[1]})

                print("\nCentroids: Piezo  Verification Input: Audio "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[2], EER_threshes[2], EER_FARs[2], EER_FRRs[2]))
                wandb.log({'epoch': epoch, 'EER/C_P_VI_A': EERs[2], 'FAR/C_P_VI_A': EER_FARs[2], 'FRR/C_P_VI_A': EER_FRRs[2]})

                print("\nCentroids: Audio  Verification Input: Piezo "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[3], EER_threshes[3], EER_FARs[3], EER_FRRs[3]))
                wandb.log({'epoch': epoch, 'EER/C_A_VI_P': EERs[3], 'FAR/C_A_VI_P': EER_FARs[3], 'FRR/C_A_VI_P': EER_FRRs[3]})

                print("\nCentroids: Audio + Piezo  Verification Input: Audio + Piezo "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[4], EER_threshes[4], EER_FARs[4], EER_FRRs[4]))
                wandb.log({'epoch': epoch, 'EER/C_AP_VI_AP': EERs[4], 'FAR/C_AP_VI_AP': EER_FARs[4], 'FRR/C_AP_VI_AP': EER_FRRs[4]})

                print("\nCentroids: Audio + Piezo  Verification Input: Piezo + Audio "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[5], EER_threshes[5], EER_FARs[5], EER_FRRs[5]))
                wandb.log({'epoch': epoch, 'EER/C_AP_VI_PA': EERs[5], 'FAR/C_AP_VI_PA': EER_FARs[5], 'FRR/C_AP_VI_PA': EER_FRRs[5]})
                
                print("\nCentroids: X  Verification Input: X "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[-1], EER_threshes[-1], EER_FARs[-1], EER_FRRs[-1]))
                wandb.log({'epoch': epoch, 'EER/C_X_VI_X': EERs[-1], 'FAR/C_X_VI_X': EER_FARs[-1], 'FRR/C_X_VI_X': EER_FRRs[-1]})


    return model_audio, model_piezo, converter


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # initialize the wandb configuration
    time_stamp = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
    wandb.init(
        # team name
        entity="piezobuds",

        # set the project name
        project="PiezoBuds",

        # params of the task
        # TODO: add configuration summaray to the time stamp
        name=time_stamp
    )
    drawer = UniqueDraw()

    ids = list(range(22))
    valid_ids = drawer.draw_numbers(22, 4)
    train_ids = [item for item in ids if item not in valid_ids]
    print(train_ids)
    print(valid_ids)

    if len(sys.argv) != 5:
        # model = LSTMWithAttention(256, 128, 2, 128).to(device)
        model_piezo = extractor1DCNN(channel_number=160,
                               in_feature=256,
                               out_feature=128,
                               device=device).to(device)
        model_audio = extractor1DCNN(channel_number=160,
                               in_feature=256,
                               out_feature=128,
                               device=device).to(device)
        is_CNN = False
        num_of_epoches = 4000
        store_path = 'test.pth'
        lr = 0.001
    else:
        if sys.argv[1] == '0':
            # CNN
            model = CNNModel(128).to(device)
            is_CNN = True
        else:
            # LSTM
            model = LSTMWithAttention(256, 128, 3, 128).to(device)
            is_CNN = False
        num_of_epoches = int(sys.argv[2])
        store_path = sys.argv[3]
        lr = float(sys.argv[4])
    converter = ConverterNetwork2D21D(128).to(device)
    # converter.apply(ConverterNetwork1DCNN.weights_init_xavier)
    
    train_set = STFTFeatureSet(False, train_ids, 30)
    test_set = STFTFeatureSet(False, valid_ids, 30)
    ge2e_loss = GE2ELoss(device)
    optimizer = torch.optim.Adam([
        {'params': model_piezo.parameters()},
        {'params': model_audio.parameters()},
        {'params': ge2e_loss.parameters()}
    ], lr=lr)
    optimizer_converter = torch.optim.Adam([
        {'params': converter.parameters()}
    ], lr=lr)
    # optimizer_converter = torch.optim.SGD([
    #     {'params': converter.parameters()}
    # ], lr=lr, momentum=0.9, nesterov=True)
    model_audio, model_piezo, converter = train_and_test_model(device, model_audio, model_piezo, converter, ge2e_loss, is_CNN, train_set, test_set, optimizer, optimizer_converter, num_of_epoches)
    # torch.save(model_audio.state_dict(), store_path)

