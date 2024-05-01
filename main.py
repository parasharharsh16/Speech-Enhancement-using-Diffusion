from utills import *
from params import *
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
train_loader = load_data(dataset_path, "train", batch_size)

model = train(model, lr, train_loader, max_epochs, batch_size, device)
#save model
torch.save(model.state_dict(), save_model_path)