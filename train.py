import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.optim as optim
import re 

from vocabulary import TokenVocabulary
from build_dataset import BuildDataset
from transformer import Transformer
from utils import construct_mask

class CustomDataset(Dataset):
    def __init__(self, translation_pairs_tensor):
        self.translation_pairs_tensor = translation_pairs_tensor

    def __len__(self):
        return len(self.translation_pairs_tensor)

    def __getitem__(self, idx):
        X = self.translation_pairs_tensor[idx][0]
        y = self.translation_pairs_tensor[idx][1]

        return X, y


def prepare_dataloader(CustomDataset, dataset, batch_size=128, train_val_split=0.2):
    custom_dataset = CustomDataset(dataset.sequence)

    n_samples = len(custom_dataset)
    split_ix = int(n_samples * train_val_split)
    
    train_indices, val_indices = np.arange(split_ix), np.arange(split_ix, n_samples)
    train_dataloader = DataLoader(
            custom_dataset, sampler=SubsetRandomSampler(train_indices), batch_size=batch_size
            )
    val_dataloader = DataLoader(
            custom_dataset, sampler=SubsetRandomSampler(val_indices), batch_size=batch_size
            )

    return train_dataloader, val_dataloader

def train(model, device, train_dataloader, criterion, optimizer, clip):
    epoch_loss = 0
    for i, (X_batch, y_batch) in enumerate(train_dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch, y_batch[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        output = output.to(device)
        y_batch = y_batch[:, 1:].contiguous().view(-1)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss

def predict_test_text(text_batch, dataset, model):
    tokenized_sentences = []
    for sentence in text_batch:
        tokens = re.findall(r"\w+|[^\s\w]+", sentence)
        tokenized_sentences.append(tokens)

    max_length = max([len(tokens) for tokens in tokenized_sentences])
    tokenized_sentences = [
        t + ((max_length - len(t)) * [dataset.pad_token])
        for t in tokenized_sentences
    ] 

    encoder_input = torch.empty((0, max_length + 2), dtype=torch.long)
    for t in tokenized_sentences:
        indexed_tokens = torch.LongTensor(
                [[dataset.ch2ix[dataset.init_token]]
                + [dataset.ch2ix[token] for token in t]
                + [dataset.ch2ix[dataset.eos_token]]]
            )
        encoder_input = torch.cat((encoder_input, indexed_tokens), dim=0)

    # to start with prediction we need to pass an initial decoder start token
    decoder_input = torch.LongTensor(
                    [[dataset.ch2ix[dataset.init_token]]]
            )
    
    # we will predict first 10 tokens
    max_decode_len = 10
    source_mask, target_mask = construct_mask(encoder_input, decoder_input, dataset.ch2ix[dataset.pad_token], dataset.ch2ix[dataset.pad_token])
    # first get output from encoder part 
    with torch.no_grad():
        encoder_output = model.encoder(
                encoder_input, source_mask
            )

    for i in range(max_decode_len):
        with torch.no_grad():
            decoder_output = model.decoder(
                    decoder_input,
                    encoder_output,
                    target_mask,
                    source_mask,
                )
        # take token with maximum probability
        predicted_token = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)
        decoder_input = torch.cat((decoder_input, predicted_token), dim=-1) # add the predicted token to the existing decoder input
        source_mask, target_mask = construct_mask(encoder_input, decoder_input, dataset.ch2ix[dataset.pad_token], dataset.ch2ix[dataset.pad_token])
    
    # use decoder input to iterate as it has the predicted tokens 
    for output in decoder_input:
        output_string = ''
        for index in output[1:]:
            character = dataset.ix2ch[index.item()]
            output_string = output_string + character + ' '
        print(output_string)

dataset = BuildDataset(TokenVocabulary)

train_dataloader, val_dataloader = prepare_dataloader(CustomDataset, dataset)

device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

model = Transformer(source_dim=dataset.vocab_size,
                target_dim=dataset.vocab_size,
                max_len=dataset.max_len,
                source_pad_idx=dataset.ch2ix[dataset.pad_token],
                target_pad_idx=dataset.ch2ix[dataset.pad_token],
                embedding_dim=128,
                dropout_p=0.5,
                device=device).to(device)
model.train()

LEARNING_RATE = 0.0001
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.ch2ix[dataset.pad_token])
clip = 1
epochs = 10
predict_text_batch = ['This is going to be easy and the model will generate a prediction for this sentence.']

for i in range(epochs):
    print("Epoch: {}/{}".format(i+1, train(model, device, train_dataloader, criterion, optimizer, clip)))
    # predict_test_text(predict_text_batch, dataset, model)