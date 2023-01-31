import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from Dataset import get_loader
from Model import CNNtoRNN

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="/home/rmarefat/projects/github/image_captioning/Flicker8k_Dataset",
        annotation_file="/home/rmarefat/projects/github/image_captioning/en_farsi_captions.txt",
        transform=transform,
        num_workers=2,
        batch_size=512
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # device  = 'cpu'
    load_model = False
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    # vocab_size = len(dataset.vocab)
    # print("vocab_size", vocab_size)
    vocab_size_en = 1867
    vocab_size_fa = 1722

    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size_en, vocab_size_fa, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi_en["<PAD>"])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()
    running_loss = []
    for epoch in range(num_epochs):
        # Uncomment the line below to see a couple of test cases
        # print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, en_captions, fa_captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            en_captions = en_captions.to(device)
            fa_captions = fa_captions.to(device)

            en_outputs, fa_outputs = model(imgs, en_captions[:-1], fa_captions[:-1])

            en_loss = criterion(
                en_outputs.reshape(-1, en_outputs.shape[2]), en_captions.reshape(-1)
            )
            fa_loss = criterion(
                fa_outputs.reshape(-1, fa_outputs.shape[2]), fa_captions.reshape(-1)
            )


            loss = en_loss + fa_loss

            running_loss.append(loss.item())
            
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            

            torch.save(model.state_dict(), f"/home/rmarefat/projects/github/image_captioning/weights/{epoch}.pt")

        avg_loss = round(sum(running_loss) / len(running_loss), 3)
        print(f"Epoch: {epoch} | Loss: {avg_loss}")

if __name__ == "__main__":
    train()