from torch.utils.data import dataloader
from dataprep import *
from model import *
from torch import nn
import math
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import soundfile as sf
#function to load the data to dataloder
def load_data(dataset_path, mode, batch_size):
    dataset = VCTKDataset(dataset_path, mode, sample_size=sample_size, train_ratio=train_ratio)
    data_loader = dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

#function to train the diffusion model
def compute_mse(spectrogram1, spectrogram2):
    # Ensure both spectrograms have the same shape
    assert spectrogram1.shape == spectrogram2.shape, "Spectrograms must have the same shape"
    
    # Compute mean squared error
    mse = np.mean((spectrogram1 - spectrogram2)**2)
    
    return mse

def get_loss(model, noise, clean, t, device):
    clean_stft, _ = get_stft(clean)
    noise_stft, _ = get_stft(noise)
    #clean, noisy = forwad_diff(clean_stft, noise_stft, t, device)
    clean_stft = torch.tensor(clean_stft).to(device)
    noisy_stft = torch.tensor(noise_stft).to(device)
    clean = clean_stft.unsqueeze(1)
    noisy = noisy_stft.unsqueeze(1)
    
    t= t.to(device)
    pred = model(noisy, t)
    
    return F.l1_loss(clean, pred),pred

#traing loop
def train(model,learning_rate,train_loader,epochs,batch_size,device):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    wandb.init(
    # set the wandb project where this run will be logged
    project="Speech Enhancement using Diffusion",
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "U-NET",
    "dataset": "VCTK-Corpus",
    "Noise": "Environmental Noise",
    "epochs": epochs,
    "batch_size": batch_size,
    "train_ratio": train_ratio,
    "sample_size": f"{sample_size*100}%",
    "sample_items": len(train_loader)
    }
    )
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Training Epochs {epoch+1}")
        for batch in pbar:
            clean, noise = batch
            batch_size = clean.shape[0]
            optimizer.zero_grad()
            t = torch.randint(0, 10, (batch_size,), device='cpu').long()
            loss,_ = get_loss(model, noise,clean, t,device)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        wandb.log({"Train loss": loss.item()})
    return model

#evaluate model
def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()
    total_loss = 0
    predictions = []
    inputwave = []
    eval_pbar = tqdm(test_loader, desc="Evaluating")
    count = 1
    with torch.no_grad():
        for batch in eval_pbar:
            clean, noise = batch
            batch_size = clean.shape[0]
            t = torch.randint(0, 10, (batch_size,), device='cpu').long()
            loss,pred = get_loss(model, noise,clean, t,device)
            total_loss += loss.item()
            predictions.append(pred)
            inputwave.append(noise)
    return predictions,inputwave, total_loss/len(test_loader)

#visualize the list of specogram
def visualize_spectogram(spectograms):
    spectrogram_shape = spectograms[0].shape
    # Calculate the figure size based on the spectrogram size

    fig, axs = plt.subplots(1, len(spectograms), figsize=(20, 10))
    for i, spectrogram in enumerate(spectograms):
        spectrogram = spectrogram.squeeze(0).squeeze(0)
        spectrogram = spectrogram.detach().cpu().numpy()
        axs[i].imshow(spectrogram, aspect='auto', origin='lower')
        axs[i].set_title(f'Spectrogram {i+1}')
    plt.savefig('output/spectogram.png')
    plt.show()



def get_stft(waveform):
    if not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform)
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)
    magnitude = magnitude[:,:1024, :64]
    phase = phase[:,:1024, :64]
    magnitude = np.pad(magnitude, ((0, 0), (0, 0), (0, 1)))
    phase = np.pad(phase, ((0, 0), (0, 0), (0, 1)))
    return magnitude, phase

#Regerate the wave from the predicted stft
def invert_stft_to_audio(generated, input_wave,n_fft = 2048, hop_length = 512):
    if not isinstance(input_wave, np.ndarray):
        input_wave = np.array(input_wave)
    if not isinstance(generated, np.ndarray):
        generated = np.array(generated)
    _,phase = get_stft(input_wave)

    waveform_reg = librosa.istft(generated * phase, hop_length=hop_length)
    return waveform_reg

def save_samples(predictions,inputwave):
    count = 0
    for i, pred in enumerate(predictions):
        pred = pred.squeeze(0).squeeze(0)
        pred = pred.detach().cpu().numpy()
        waveform_reg = invert_stft_to_audio(pred,inputwave[i].detach().numpy(), hop_length=512)
        sf.write(f'output/sample_{i}.wav', waveform_reg.squeeze(0), 16000)
        if count == 10:
            break
        count += 1
