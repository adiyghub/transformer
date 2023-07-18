from vocabulary import TokenVocabulary
import torch
import re
import torch.nn.functional as F

def convert_source_target_list_to_tokens(
    source_target_pair, vocab_index, init_token, eos_token
):
    tensors = []
    for pair in source_target_pair:
        tensor_pair = []
        for sentence in pair:
            tensor = torch.LongTensor(
                [vocab_index[init_token]]
                + [vocab_index[ch] for ch in re.findall(r"\w+|[^\s\w]+", sentence)]
                + [vocab_index[eos_token]]
            )
            tensor_pair.append(tensor)
        tensors.append(tensor_pair)
    return tensors


class BuildDataset:
    def __init__(
        self, TokenVocabulary, source_file="train.en", target_file="train.de"
    ):
        self.token_vocab = TokenVocabulary()
        self.pad_token = self.token_vocab.pad_token
        self.init_token = self.token_vocab.init_token
        self.eos_token = self.token_vocab.eos_token
        self.source_file = source_file
        self.target_file = target_file

        self.token_vocab.build_vocabulary(self.source_file, self.target_file)
        
        self.ch2ix = self.token_vocab.ch2ix
        self.ix2ch = self.token_vocab.ix2ch
        self.vocab_size = len(set(self.token_vocab.ch2ix.values())) 
         
        self.tokenized_sentences = convert_source_target_list_to_tokens(
            self.token_vocab.source_target_list,
            self.token_vocab.ch2ix,
            self.token_vocab.init_token,
            self.token_vocab.eos_token,
        )
        self.max_len = max([max([len(sentence[0]) for sentence in self.tokenized_sentences]), 
                    max([len(sentence[1]) for sentence in self.tokenized_sentences])])

        padded_sources = [
            F.pad(src, (0, self.max_len  - src.size(0)), value=self.ch2ix[self.token_vocab.pad_token]) for src, _ in self.tokenized_sentences
            ]
        
        padded_targets = [
            F.pad(tgt, (0, self.max_len  - tgt.size(0)), value=self.ch2ix[self.token_vocab.pad_token])
            for _, tgt in self.tokenized_sentences
            ]
            
        self.sequence = torch.stack([torch.stack(padded_sources), torch.stack(padded_targets)], dim=1)

        

        
    

        

        
    
