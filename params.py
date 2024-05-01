sampling_rate = 16000
dataset_path = "data/VCTK-Corpus/wav48"
mode = "train"  #'train' #'test'
batch_size = 8
num_workers = 1
max_epochs = 20
sample_size = 0.2
train_ratio = 0.8
lr = 0.00001
n_fft = 2048
hop_length = 512
save_model_path = f"models/model_{sample_size}.pth"
save_sample_path = "sample"
T = 5
noise_path = "data/noise"
