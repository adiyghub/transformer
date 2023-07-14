import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.optim as optim

from vocabulary import TokenVocabulary
from build_dataset import BuildDataset
from transformer import Transformer

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

def train(model, train_dataloader, criterion, optimizer, clip):
    epoch_loss = 0
    for i, (X_batch, y_batch) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(X_batch, y_batch[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        y_batch = y_batch[:, 1:].contiguous().view(-1)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss

dataset = BuildDataset(TokenVocabulary)

train_dataloader, val_dataloader = prepare_dataloader(CustomDataset, dataset)

model = Transformer(source_dim=dataset.vocab_size,
                target_dim=dataset.vocab_size,
                max_len=dataset.max_len,
                source_pad_idx=dataset.ch2ix[dataset.pad_token],
                target_pad_idx=dataset.ch2ix[dataset.pad_token],
                embedding_dim=128)
model.train()

LEARNING_RATE = 0.0001
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.ch2ix[dataset.pad_token])
clip = 1
epochs = 10

for i in range(epochs):
    print("Epoch: {}/{}".format(i+1, train(model, train_dataloader, criterion, optimizer, clip)))