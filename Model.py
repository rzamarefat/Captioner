import torch
from torchvision import models

class Backbone(torch.nn.Module):
    def __init__(self, embed_size):
        super(Backbone, self).__init__()
        self.embed_size = 256
        self.hidden_size = 256

        self.inception = models.resnext101_32x8d(pretrained=False)
        self.inception.fc = torch.nn.Linear(self.inception.fc.in_features, self.embed_size)
        self.relu = torch.nn.ReLU()
        self.times = []
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features))


class DecoderRNN(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()

        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, features, captions):
        
        print("---->", captions)
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs


class CNNtoRNN(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size_en, vocab_size_fa, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = Backbone(embed_size)
        self.english_decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size_en, num_layers)
        self.persian_decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size_fa, num_layers)

    def forward(self, images, en_captions, fa_captions):
        features = self.encoderCNN(images)
        en_outputs = self.english_decoderRNN(features, en_captions)
        fa_outputs = self.persian_decoderRNN(features, fa_captions)
        return en_outputs, fa_outputs

    def caption_image(self, image, vocabulary, max_length=50, language="en"):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeecaptionsze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]



if __name__ == "__main__":
    data = torch.randn((2, 3, 356, 356))
    
    model = CNNtoRNN(256, 256, 100, 1)
    en_out, fa_out = model(data, None)
    print(out.shape)

        

