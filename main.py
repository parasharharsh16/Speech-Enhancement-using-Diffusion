from utills import *
from params import *
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
train_loader = load_data(dataset_path, "train", batch_size)

if mode == "train":
    model = train(model, lr, train_loader, max_epochs, batch_size, device)
    #save model
    torch.save(model.state_dict(), save_model_path)
else:
    model.load_state_dict(torch.load(save_model_path))

test_loader = load_data(dataset_path, "test", batch_size=1)
predictions,inputwave, test_loss = evaluate(model, test_loader, device)
print(f"Test Loss: {test_loss}")
#save samples
save_samples(predictions,inputwave)

#save sample
#convert_to_audio(predictions,hop_length=512)

