import torch
from Model import CNNtoRNN
from torchvision import transforms as T
from PIL import Image
from Dataset import Vocabulary


def infer(path_to_image=None):
    device = 'cpu'
    embed_size = 256
    hidden_size = 256
    vocab_size_en = 1867
    vocab_size_fa = 1722
    num_layers = 1
    
    path_to_image = "/home/rmarefat/projects/github/image_captioning/Flicker8k_Dataset/47871819_db55ac4699.jpg"

    transform = T.Compose(
        [T.Resize((224, 224)), T.ToTensor(),]
    )

    img = Image.open(path_to_image)
    img = transform(img)

    img = img.to(device)
    img = torch.unsqueeze(img, dim=0)
    print("img.shape", img.shape)

    model = CNNtoRNN(embed_size, hidden_size, vocab_size_en, vocab_size_fa, num_layers).to(device)
    model.load_state_dict(torch.load("./weights/35.pt"))

    with open("./en_farsi_captions.txt") as h:
        content = [f.replace("\n", "") for f in sorted(h.readlines())]
    en_captions = [c.split("|||")[2] for c in content]
    fa_captions = [c.split("|||")[1] for c in content]

    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(en_captions, fa_captions)

    model.eval()
    en_cap, fa_cap = model.caption_image(img, vocab, max_length=50)

    print(en_cap)
    print(fa_cap)



if __name__ == "__main__":
    infer()