import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
import librosa
import numpy as np
import os
import pandas as pd
from params import *
#Reading all the available files from the dataset folder
class VCTKDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_path, mode,sample_size =0.01 ,train_ratio=0.8):
        self.dataset_path = dataset_path
        self.mode = mode
        self.files = self.get_files(self.dataset_path)
        self.files = self.files[:int(sample_size*len(self.files))]

        print(f"Total files found: {len(self.files)}")
        self.noise_path_list = self.get_noise(noise_path)
        #splitting the dataset into train and test
        if self.mode == 'train':
            self.files = self.files[:int(train_ratio*len(self.files))]
        else:
            self.files = self.files[int((train_ratio)*len(self.files)):]
        
        print(f"{mode} dataset length: {len(self.files)}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        waveform, noise = self.preprocess_audio(file_path,self.noise_path_list)
        return waveform, noise
    
    def get_files(self,dataset_path):
        files = []
        for root, _, folder_name in os.walk(dataset_path):
            for filename in folder_name:
                if filename.endswith('.wav'):
                    files.append(os.path.join(root, filename))
        return files
    def get_noise(self,noise_path):
        noise_path_list = []
        for root, _, filenames in os.walk(noise_path):
            for filename in filenames:
                noise_path_list.append(os.path.join(root, filename))
        return noise_path_list
    def preprocess_audio(self, audio_path,noise_path_list):
        # Load the audio file
        waveform, _ = librosa.load(audio_path, sr=sampling_rate)
        
        # Load a random noise file
        random_noise = np.random.choice(noise_path_list)
        noise, _ = librosa.load(random_noise, sr=sampling_rate)
        
        # Define desired length (2 seconds)
        desired_length = sampling_rate * 2
        
        # If the waveform is shorter than the desired length, pad it with zeros
        if len(waveform) < desired_length:
            padding = desired_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), 'constant')
        
        # If the noise is shorter than the desired length, pad it with zeros
        if len(noise) < desired_length:
            padding = desired_length - len(noise)
            noise = np.pad(noise, (0, padding), 'constant')
        
        # Take the middle 2 seconds of the waveform
        start = (len(waveform) - desired_length) // 2
        waveform = waveform[start:start + desired_length]
        
        # If the noise is longer than 2 seconds, take the first 2 seconds
        if len(noise) > desired_length:
            noise = noise[:desired_length]
        
        return waveform, noise




def forwad_diff(clean, noise,t,device):
    #convert to spectogram
    mel_spectrogram = MelSpectrogram(n_mels=64, n_fft=2048)
    clean_spectogran = mel_spectrogram(clean.clone().detach().unsqueeze(0)).squeeze(0)
    #clean_spectogran = mel_spectrogram(torch.tensor(clean).unsqueeze(0)).squeeze(0)
    # Adding noise in steps to make diffusion data
    t = t.view(clean.shape[0],1)
    waveform= clean + t*noise/10
    #noisy = mel_spectrogram(torch.tensor(waveform).unsqueeze(0)).squeeze(0)
    noisy = mel_spectrogram(waveform.clone().detach().unsqueeze(0)).squeeze(0)

    clean_spectogran = clean_spectogran.to(device)
    noisy = noisy.to(device)
    return clean_spectogran, noisy





#test the data
# load a audio
#waveform to wav file
#librosa.output.write_wav('output/sample.wav', waveform, sr=sampling_rate)

# Test dataset
# dataset = VCTKDataset(dataset_path, "train")
# test_dataset = VCTKDataset(dataset_path, "test")

# print("Completed")