from utills import *
from params import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
train_loader = load_data(dataset_path, "train", batch_size)

if mode == "train":
    model = train(model, lr, train_loader, max_epochs, batch_size, device)
    # save model
    torch.save(model.state_dict(), save_model_path)
else:
    model.load_state_dict(torch.load(save_model_path))

test_loader = load_data(dataset_path, "test", batch_size=1)

results,simlarity_score,similarity_noisy,test_loss = evaluate(model, test_loader, device)

print(f"Test Loss: {test_loss}")
print(f"Similarity Score: {simlarity_score}")

#save samples
save_samples_and_spectogram(results)