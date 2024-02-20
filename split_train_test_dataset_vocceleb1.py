import os
import shutil

if __name__=='__main__':

    data_file_dir = '/mnt/hdd/gen/processed_data/wav_clips_1000ms/voxceleb1/audio/'

    target_file_dir_train = '/mnt/hdd/gen/processed_data/voxceleb1/wav_clips_1000ms/train/'
    target_file_dir_test = '/mnt/hdd/gen/processed_data/voxceleb1/wav_clips_1000ms/test/'
    os.makedirs(target_file_dir_train, exist_ok=True)
    os.makedirs(target_file_dir_test, exist_ok=True)

    for i in range(10001, 11252):
        lst = os.listdir(data_file_dir + '{}/'.format(i))
        n_uttr = len(lst)
        print('User: %d' % i)
        print(n_uttr)
        n_uttr_test = int(n_uttr * 0.2)
        if n_uttr_test < 80:
            n_uttr_test = 80
        n_uttr_train = n_uttr - n_uttr_test
        print('Train: %d, Test: %d' % (n_uttr_train, n_uttr_test))
        
        for j in range(0, n_uttr_train):
            audio_path = target_file_dir_train + '{}/'.format(i)
            os.makedirs(audio_path, exist_ok=True)

            source_audio = data_file_dir + '{}/'.format(i) + lst[j]
            target_audio = audio_path + '{}.wav'.format(j)
            # print('Copy ' + source_audio + ' to ' + target_audio)
            shutil.copyfile(source_audio, target_audio)

        for j in range(n_uttr_train, n_uttr):
            audio_path = target_file_dir_test + '{}/'.format(i)
            os.makedirs(audio_path, exist_ok=True)

            source_audio = data_file_dir + '{}/'.format(i) + lst[j]
            target_audio = audio_path + '{}.wav'.format(j - n_uttr_train)
            # print('Copy ' + source_audio + ' to ' + target_audio)
            shutil.copyfile(source_audio, target_audio)



        # n_uttr = 

