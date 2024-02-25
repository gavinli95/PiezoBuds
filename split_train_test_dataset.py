import os
import shutil

if __name__=='__main__':
    n_user = 70

    data_file_dir = '/mnt/hdd/gen/processed_data/wav_clips_500ms/piezobuds/'

    target_file_dir_train = '/mnt/hdd/gen/processed_data/wav_clips_500ms/piezobuds_new/train/'
    target_file_dir_test = '/mnt/hdd/gen/processed_data/wav_clips_500ms/piezobuds_new/test/'
    os.makedirs(target_file_dir_train, exist_ok=True)
    os.makedirs(target_file_dir_test, exist_ok=True)

    for i in range(n_user):
        lst = os.listdir(data_file_dir + 'piezo/{}/'.format(i))
        n_uttr = len(lst)
        print('User: %d' % i)
        print(n_uttr)
        n_uttr_test = int(n_uttr * 0.2)
        if n_uttr_test < 100:
            n_uttr_test = 100
        n_uttr_train = n_uttr - n_uttr_test
        print('Train: %d, Test: %d' % (n_uttr_train, n_uttr_test))
        
        for j in range(0, n_uttr_train):
            piezo_path = target_file_dir_train + 'piezo/{}/'.format(i)
            audio_path = target_file_dir_train + 'audio/{}/'.format(i)
            os.makedirs(piezo_path, exist_ok=True)
            os.makedirs(audio_path, exist_ok=True)

            source_piezo = data_file_dir + 'piezo/{}/'.format(i) + lst[j]
            target_piezo = piezo_path + '{}.wav'.format(j)
            print('Copy ' + source_piezo + ' to ' + target_piezo)
            shutil.copyfile(source_piezo, target_piezo)

            source_audio = data_file_dir + 'audio/{}/'.format(i) + lst[j]
            target_audio = audio_path + '{}.wav'.format(j)
            print('Copy ' + source_audio + ' to ' + target_audio)
            shutil.copyfile(source_audio, target_audio)

        for j in range(n_uttr_train, n_uttr):
            piezo_path = target_file_dir_test + 'piezo/{}/'.format(i)
            audio_path = target_file_dir_test + 'audio/{}/'.format(i)
            os.makedirs(piezo_path, exist_ok=True)
            os.makedirs(audio_path, exist_ok=True)

            source_piezo = data_file_dir + 'piezo/{}/'.format(i) + lst[j]
            target_piezo = piezo_path + '{}.wav'.format(j - n_uttr_train)
            print('Copy ' + source_piezo + ' to ' + target_piezo)
            shutil.copyfile(source_piezo, target_piezo)

            source_audio = data_file_dir + 'audio/{}/'.format(i) + lst[j]
            target_audio = audio_path + '{}.wav'.format(j - n_uttr_train)
            print('Copy ' + source_audio + ' to ' + target_audio)
            shutil.copyfile(source_audio, target_audio)



        # n_uttr = 

