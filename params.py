sampling_rate = 16000
dataset_path = 'data/VCTK-Corpus/wav48'
mode = 'train'
batch_size = 16
num_workers = 1
max_epochs = 20
sample_size = 0.20
train_ratio = 0.8
lr = 0.001
save_model_path = f'models/model_{sample_size}.pth'
save_sample_path = 'sample'

noise_path = 'data/noise'