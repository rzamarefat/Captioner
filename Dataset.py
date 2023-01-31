import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms

spacy_eng = spacy.load("en")


class Vocabulary:
    def __init__(self, freq_threshold):

        self.itos_en = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi_en = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        self.itos_fa = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi_fa = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        self.freq_threshold = freq_threshold

    def __len__(self):
        print(len(self.itos_en))
        print(len(self.itos_fa))
        print(len(self.stoi_en))
        print(len(self.stoi_fa))
        return len(self.itos_en)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    @staticmethod
    def tokenizer_fa(text):
        return [tok.lower() for tok in text.split(" ")]

    def build_vocabulary(self, en_sentence_list, fa_sentence_list):
        en_frequencies = {}
        fa_frequencies = {}
        idx = 4

        for sentence in en_sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in en_frequencies:
                    en_frequencies[word] = 1
                else:
                    en_frequencies[word] += 1

                if en_frequencies[word] == self.freq_threshold:
                    self.stoi_en[word] = idx
                    self.itos_en[idx] = word
                    idx += 1

        idx = 4
        for sentence in fa_sentence_list:
            for word in self.tokenizer_fa(sentence):
                if word not in fa_frequencies:
                    fa_frequencies[word] = 1
                else:
                    fa_frequencies[word] += 1

                if fa_frequencies[word] == self.freq_threshold:
                    self.stoi_fa[word] = idx
                    self.itos_fa[idx] = word
                    idx += 1



        print("len(self.stoi_fa)", self.stoi_fa)

    def numericalize(self, text, language):

        if language == "en":
            tokenized_text = self.tokenizer_eng(text)

            return [
                self.stoi_en[token] if token in self.stoi_en else self.stoi_en["<UNK>"] for token in tokenized_text
            ]

        elif language == "fa":
            tokenized_text = self.tokenizer_fa(text)

            return [
                self.stoi_fa[token] if token in self.stoi_fa else self.stoi_fa["<UNK>"] for token in tokenized_text
                ]

        else:
            raise ValueError("Provide a valid language: en/fa")


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5, language=None):
        self.root_dir = root_dir
        

        
        self.captions_file = captions_file
        with open(self.captions_file) as h:
            content = [f.replace("\n", "") for f in sorted(h.readlines())]

        self.imgs = [os.path.join(self.root_dir ,c.split("|||")[0].split("____")[0]) for c in content]
        self.en_captions = [c.split("|||")[2] for c in content]
        self.fa_captions = [c.split("|||")[1] for c in content]

        self.transform = transform

        
        self.vocab = Vocabulary(freq_threshold)
        
        self.vocab.build_vocabulary(self.en_captions, self.fa_captions)

    def __len__(self):
        return len(self.en_captions)

    def __getitem__(self, index):
        en_caption = self.en_captions[index]
        fa_caption = self.fa_captions[index]

        img = self.imgs[index]
        img = Image.open(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption_en = [self.vocab.stoi_en["<SOS>"]]
        numericalized_caption_en += self.vocab.numericalize(en_caption, language="en")
        numericalized_caption_en.append(self.vocab.stoi_en["<EOS>"])


        numericalized_caption_fa = [self.vocab.stoi_en["<SOS>"]]
        numericalized_caption_fa += self.vocab.numericalize(fa_caption, language="fa")
        numericalized_caption_fa.append(self.vocab.stoi_en["<EOS>"])

        return img, torch.tensor(numericalized_caption_en), torch.tensor(numericalized_caption_fa)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        en_targets = [item[1] for item in batch]
        fa_targets = [item[2] for item in batch]
        en_targets = pad_sequence(en_targets, batch_first=False, padding_value=self.pad_idx)
        fa_targets = pad_sequence(fa_targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, en_targets, fa_targets


def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=1,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi_en["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        "/home/rmarefat/projects/github/image_captioning/Flicker8k_Dataset",
         "/home/rmarefat/projects/github/image_captioning/en_farsi_captions.txt", 
         transform=transform
    )

    


    for idx, (imgs, en_captions, fa_captions) in enumerate(loader):
        # print(imgs.shape)
        # print(en_captions)
        # print(fa_captions)

        
        break