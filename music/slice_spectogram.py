"""Slice the spectogram into multiple 128x128 images which will be the input to the 
Convolutional Neural Network.

The function has the side-effect of generating respective `genre_variable` and
`song_variable` related folders, with the respective spectograms being extracted
from the raw images.

Each image is subsampled according to the subsample_size (default 128x128 pixels).
These subsamples act as multiple data-points for the same genre/song
and are saved inside the folder with appropriate naming based on this.  
"""
import os

# ----- Vikram-Shenoy's method -------
def vs_method():    
    import re
    from PIL import Image
    
    def slice_spect(verbose=0, mode=None):
        if mode == 'Train':
            if os.path.exists('data_collection/data/Train_Sliced_Images'):
                return

            image_folder = 'Train_Spectogram_Images'
            filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                if f.endswith('.jpg')]
            
            counter = 0
            if verbose > 0:
                print("Slicing Spectograms ...")
            if not os.path.exists('Train_Sliced_Images'):
                os.makedirs('Train_Sliced_Images')
            for f in filenames:
                genre_variable = re.search('Train_Spectogram_Images/.*_(.+?).jpg', f).group(1)
                img = Image.open(f)
                subsample_size = 128
                width, height = img.size
                number_of_samples = width // subsample_size
                for i in range(number_of_samples):
                    start = i * subsample_size
                    img_temporary = img.crop((start, 0., start + subsample_size, subsample_size))
                    img_temporary.save('Train_Sliced_Images/' + str(counter) + "_" + genre_variable + ".jpg")
                    counter += 1
        
        elif mode == 'Test':
            if os.path.exists('Test_Sliced_Images'):
                return

            image_folder = 'Test_Spectogram_Images'
            filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                if f.endswith('.jpg')]
            
            counter = 0
            if verbose > 0:
                print("Slicing Spectograms ...")
            if not os.path.exists('Test_Slices_Images'):
                os.makedirs('Test_Sliced_Images')
            for f in filenames:
                song_variable = re.search('Test_Spectogram_Images/(.+?).jpg', f).group(1)
                img = Image.open(f)
                subsample_size = 128
                width, _ = img.size
                number_of_samples = width // subsample_size
                for i in range(number_of_samples):
                    start = i * subsample_size
                    img_temporary = img.crop((start, 0., start + subsample_size, subsample_size))
                    img_temporary.save('Test_Sliced_Images/' + str(counter) + "_" + song_variable + ".jpg")
                    counter += 1
                
        return
    
    return slice_spect

# -------- Random Kaggle method --------
def kaggle_method(verbose=0, mel_spectogram=True):
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    
    audio_fpath = r'data_collection/data'
    audio_clips = os.listdir(audio_fpath)
    print('Number of .wav/.mp3 files in audio folder = ', len(audio_clips))

    for i in range(len(audio_clips)):
        x, sr = librosa.load(audio_fpath + audio_clips[i], sr=44100)
        
        # NOTE: Uncomment below for first 3 cases to show the individual plots
        if i < 3:
            print(type(x), type(sr))
            print(x.shape, sr)
            plt.figure(figsize=(14, 5))
            librosa.display.waveshow(x, sr)

        # Convert the audio waveform to spectogram
        if mel_spectogram:
            # TODO: correctly set the parameters below
            X = librosa.feature.melspectrogram(x, sr=sr, n_mels=20, n_fft=20)
        else:
            X = librosa.stft(x) # NOTE: Or we should have librosa.feature.melspectogram(y, sr=sr, n_mels=?, fmax=?)
        
        Xdb = librosa.amplitude_to_db(abs(X))
        
        image_name = "data_collection/data/Spectograms"

        plt.figure(figsize=(14, 5))
        img_time_hertz = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
        current_name = ''.join([image_name, 'Image_Time_Hertz_{}'.format(str(i+1))])
        plt.savefig(current_name, img_time_hertz)
        if verbose and i < 3:
            plt.colorbar()

        # Apply log transformation on the loaded audio signals
        plt.figure(figsize=(14, 5))
        img_time_log_hertz = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
        current_name = ''.join([image_name, 'Image_Time_Log_Hertz_{}'.format(str(i+1))])
        plt.savefig(current_name, img_time_log_hertz)        
        if verbose and i < 3:
            plt.colorbar()