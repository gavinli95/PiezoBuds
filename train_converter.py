import sys
import time
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import STFTFeatureSet
from my_models import CNNModel, LSTMWithAttention, GE2ELoss, ConverterNetwork, extractor1DCNN, Converter_w_triplet, Converter_ALAH2X
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


def train_and_test_model(device, extractor, converter, ge2e_loss, is_cnn, train_set, test_set, optimizer, optimizer_converter, num_epochs=2000):
    writer = SummaryWriter()
    # number of user for train and test in each epoch. Train 4 users, test 2 users.
    batch_size = 2
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=True)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # train and test extractor
        # for phase in ['train', 'test']:
        for phase in ['train']:
            if phase == 'train':
                # set extractor to training
                extractor.train()
                converter.train()
                dataloader = train_loader
            else:
                # set extractor to test
                extractor.eval()
                converter.eval()
                dataloader = test_loader
            
            # train each batch
            num_of_batches = 0
            loss_avg_batch_converter = 0.0

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
                        
                        # TODO: change the original audios into audios_high and audios_low
                        # piezos = piezos[:, :, :128]
                        # audios_high = audios[:, :, 128:]
                        # audios_low = audios[:, :, :128]

                        embeddings_piezo, _ = extractor(piezos)
                        embeddings_audio, _ = extractor(audios)
                        # embeddings_audio_l, _ = extractor(audios_low)

                        # convert the Ep & Eah to Ex as well as Ey
                        embeddings_x, embeddings_y = converter(embeddings_audio, embeddings_piezo)
                        _, output_dim = embeddings_piezo.shape

                        # reshape the output to (number of speaker, number of utter, length of feature)
                        embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters, output_dim))
                        embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters, output_dim))
                        # embeddings_audio_h = torch.reshape(embeddings_audio_h, (num_of_speakers, num_of_utters, output_dim))
                        embeddings_x = torch.reshape(embeddings_x, (num_of_speakers, num_of_utters, output_dim))
                        embeddings_y = torch.reshape(embeddings_y, (num_of_speakers, num_of_utters, output_dim))
                        # for x & y we change the last dim to be the half of the original embedding vectors
                        # embeddings_x = torch.reshape(embeddings_x, (num_of_speakers, num_of_utters, int(output_dim / 2)))
                        # embeddings_y = torch.reshape(embeddings_y, (num_of_speakers, num_of_utters, int(output_dim / 2)))
                        # embeddings_audio_to_piezo = torch.reshape(embeddings_audio_to_piezo, (num_of_speakers, num_of_utters, output_dim))
                        # embeddings_piezo_to_audio = torch.reshape(embeddings_piezo_to_audio, (num_of_speakers, num_of_utters, output_dim))

                        centroids_x = get_centroids(embeddings_x)
                        centroids_y = get_centroids(embeddings_y)
                        centroids_piezo = get_centroids(embeddings_piezo)
                        # centroids_y = get_centroids(embeddings_y)

                        # converter loss
                        # For loss, we have ge2e(X2Y) + ge2eloss(conc(AH, X, Y))
                        # as for the last concatenation it could be visualized as
                        #           -----------------
                        #                   AH
                        #           -----------------
                        #               X    |   Y
                        #           -----------------

                        embeddings_yp = torch.cat((embeddings_y, embeddings_piezo), dim=-1) # converter out dim * 2
                        embeddings_xp = torch.cat((embeddings_x, embeddings_piezo), dim=-1)
                        # embeddings_x2y_w_ah = torch.cat((embeddings_x2y, embeddings_audio_h), dim=-1) # converter out dim * 2 + extractor out dim
                        # loss_x, _ = ge2e_loss(embeddings_x, embeddings_x)
                        # loss_y, _ = ge2e_loss(embeddings_y, embeddings_y)
                        loss_yp, _ = ge2e_loss(embeddings_yp, embeddings_yp)
                        loss_xp, _ = ge2e_loss(embeddings_xp, embeddings_xp)
                        # combined = torch.cat((embeddings_piezo, embeddings_y, embeddings_x), dim=1)
                        # loss_converter_ge2e, _ = ge2e_loss(combined, combined)
                        # combined = torch.cat((embeddings_piezo, embeddings_y, embeddings_x), dim=-1)
                        # loss_converter_ge2e_concat, _ = ge2e_loss(combined, combined)
                        # loss_converter_ge2e_y, _ = ge2e_loss(embeddings_y, embeddings_x)

                        # sim_matrix_p2x = get_modal_cossim(embeddings_piezo, centroids_x)
                        # sim_matrix_p2y = get_modal_cossim(embeddings_piezo, centroids_y)
                        # loss_converter_p2x, loss_converter_p2x_user = cal_intra_loss(sim_matrix_p2x, device)
                        # loss_converter_p2y, loss_converter_p2y_user = cal_intra_loss(sim_matrix_p2y, device)
                        sim_matrix_x2p = get_modal_cossim(embeddings_x, centroids_piezo)
                        sim_matrix_p2x = get_modal_cossim(embeddings_piezo, centroids_x)
                        loss_converter_x2p, loss_converter_x2p_user = cal_intra_loss(sim_matrix_x2p, device)
                        loss_converter_p2x, loss_converter_p2x_user = cal_intra_loss(sim_matrix_p2x, device)
                        sim_matrix_y2p = get_modal_cossim(embeddings_y, centroids_piezo)
                        sim_matrix_p2y = get_modal_cossim(embeddings_piezo, centroids_y)
                        loss_converter_y2p, loss_converter_y2p_user = cal_intra_loss(sim_matrix_y2p, device)
                        loss_converter_p2y, loss_converter_p2y_user = cal_intra_loss(sim_matrix_p2y, device)

                        loss_converter = (loss_converter_x2p + loss_converter_p2x + loss_converter_p2y + loss_converter_y2p)/2 + (loss_yp + loss_xp)
                        # wp = 0.7
                        # loss_converter = ((loss_converter_p2x_user * wp - loss_converter_a2x_user * (1 - wp)) ** 2).sum() + 0.5 * (wp * loss_converter_p2x + (1 - wp) * loss_converter_a2x) +  2 * loss_x
                        # loss_converter = torch.square(loss_converter_a2x - loss_converter_p2x) + loss_converter_p2x + loss_converter_a2x + loss_x
                        # loss_converter = (loss_converter_ge2e_concat) + (loss_converter_p2x + loss_converter_p2y) / 2
                        # loss_converter = loss_converter_ge2e_y + loss_converter_p2x
                        loss_avg_batch_converter += loss_converter.item()

                        optimizer_converter.zero_grad()
                        loss_converter.backward()
                        torch.nn.utils.clip_grad_norm_(converter.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
                        optimizer_converter.step()

            if phase == 'train':
                epoch_loss_converter = loss_avg_batch_converter / num_of_batches
                print(f'{phase} Loss Converter: {epoch_loss_converter:.4f}')
                # writer.add_scalar('Loss/train', epoch_loss, epoch)
                wandb.log({'epoch': epoch, 'Loss/train_converter': epoch_loss_converter})
    return converter


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

    ids = list(range(42))

    # valid_ids = [4, 7, 13, 21]
    # valid_ids = [2, 8, 9, 14]
    valid_ids = [4, 7, 13, 21, 30, 36, 29, 41]
    # valid_ids = [1, 5, 15, 20]
    # valid_ids = [3, 6, 9, 14]
    # valid_ids = [3, 6, 8, 1]
    # valid_ids = [11, 12, 14, 16]
    # valid_ids = drawer.draw_numbers(22, 4)
    train_ids = [item for item in ids if item not in valid_ids]
    print(train_ids)
    print(valid_ids)

    if len(sys.argv) != 5:
        # extractor = LSTMWithAttention(256, 128, 2, 128).to(device)
        extractor = extractor1DCNN(channel_number=160,
                               in_feature=256,
                               out_feature=128,
                               device=device).to(device)
        is_CNN = False
        num_of_epoches = 2000
        store_path = 'extractor_128_4_1.pth'
        lr = 0.001
        extractor.load_state_dict(torch.load(store_path))
    else:
        if sys.argv[1] == '0':
            # CNN
            extractor = CNNModel(128).to(device)
            is_CNN = True
        else:
            # LSTM
            extractor = LSTMWithAttention(256, 128, 3, 128).to(device)
            is_CNN = False
        num_of_epoches = int(sys.argv[2])
        store_path = sys.argv[3]
        lr = float(sys.argv[4])
    converter = ConverterNetwork(256, 512).to(device)
    # converter = ConverterNetwork1DCNN()
    # converter.apply(ConverterNetwork1DCNN.weights_init_xavier)
    
    train_set = STFTFeatureSet(False, train_ids, 30)
    test_set = STFTFeatureSet(False, valid_ids, 30)
    ge2e_loss = GE2ELoss(device)
    optimizer = torch.optim.SGD([
        {'params': extractor.parameters()}
    ], lr=lr)
    optimizer_converter = torch.optim.SGD([
        {'params': converter.parameters()},
        {'params': ge2e_loss.parameters()}
    ], lr=lr, momentum=0.9, nesterov=True)
    converter = train_and_test_model(device, extractor, converter, ge2e_loss, is_CNN, train_set, test_set, optimizer, optimizer_converter, num_of_epoches)
    torch.save(converter.state_dict(), 'converter_256_512_ExEy_4.pth')

