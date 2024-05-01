from torch.utils.data import dataloader
from dataprep import *
from torch import nn
import math
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
import wandb

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



class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

def get_loss(model, noise,clean, t, device):
    clean_spectogran, noisy = forwad_diff(clean, noise,t,device)
    clean_spectogran = clean_spectogran.unsqueeze(1)
    noisy = noisy.unsqueeze(1)
    t= t.to(device)
    pred = model(noisy, t)
    
    return F.l1_loss(clean_spectogran, pred)

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
            loss = get_loss(model, noise,clean, t,device)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        wandb.log({"Train loss": loss.item()})
    return model
