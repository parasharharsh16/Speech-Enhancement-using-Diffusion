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
        print(f"Total files found: {len(self.files)}")
        self.noise_path_list = self.get_noise(noise_path)
        #splitting the dataset into train and test
        if self.mode == 'train':
            self.files = self.files[:int(train_ratio*len(self.files))]
        else:
            self.files = self.files[int((train_ratio)*len(self.files)):]
        #take sample of the dataset
        
        self.files = self.files[:int(sample_size*len(self.files))]
        print(f"{mode} dataset length: {len(self.files)}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        #diffus_data, clean_spectogram = self.prepare_diffusiondata(file_path,self.noise_path_list)
        #return diffus_data, clean_spectogram 
        waveform, noise = self.prepare_diffusiondata(file_path,self.noise_path_list)
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
    def prepare_diffusiondata(self, audio_path,noise_path_list):
        waveform, _ = librosa.load(audio_path, sr=sampling_rate)
        #mixing the audio with noise
        random_noise = np.random.choice(noise_path_list)
        noise, _ = librosa.load(random_noise, sr=sampling_rate)
        #crop audio for 1 sec
        desired_length = (sampling_rate * 1)+1

        # If the waveform is shorter than the desired length, pad it with zeros
        if len(waveform) < desired_length:
            padding = desired_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), 'constant')

        # If the noise is shorter than the desired length, pad it with zeros
        if len(noise) < desired_length:
            padding = desired_length - len(noise)
            noise = np.pad(noise, (0, padding), 'constant')

        # Crop or use the waveform and noise as is
        waveform = waveform[:desired_length]
        noise = noise[:desired_length]
        start = np.random.randint(0, len(waveform)-sampling_rate*1)
        waveform = waveform[start:start+sampling_rate*1]
        noise = noise[:sampling_rate*1]
        #convert to spectogram
        #clean_spectogran = MelSpectrogram(n_mels=64, n_fft=2048)(torch.tensor(waveform).unsqueeze(0)).squeeze(0)
        # Adding noise in steps to make diffusion data
        # data = []
        # for i in range(0, 10):
        #     waveform= waveform + noise*i/10
        #     data.append(MelSpectrogram(n_mels=64, n_fft=2048)(torch.tensor(waveform).unsqueeze(0)).squeeze(0))

        # data = np.array(data)
        return waveform, noise

def invert_spectrogram(spectrogram, hop_length):
    # Initialize a random phase
    phase = np.random.randn(*spectrogram.shape)
    
    # Iteratively improve the phase using Griffin-Lim algorithm
    for i in range(100):
        waveform = librosa.istft(spectrogram * np.exp(1j * phase), hop_length=hop_length)
        _, phase = librosa.magphase(librosa.stft(waveform, hop_length=hop_length))

    return waveform


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

# Test dataset
# dataset = VCTKDataset(dataset_path, "train")
# test_dataset = VCTKDataset(dataset_path, "test")

# print("Completed")