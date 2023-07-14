from vocabulary import LetterVocabulary
import torch
import re

def convert_source_target_list_to_tensors(
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
        self, LetterVocabulary, source_file="train.en", target_file="train.de"
    ):
        self.letter_vocab = LetterVocabulary()
        self.source_file = source_file
        self.target_file = target_file

        self.letter_vocab.build_vocabulary(self.source_file, self.target_file)
        
        self.ch2ix = self.letter_vocab.ch2ix 
        self.sentence_pairs_dataset = convert_source_target_list_to_tensors(
            self.letter_vocab.source_target_list,
            self.letter_vocab.ch2ix,
            self.letter_vocab.init_token,
            self.letter_vocab.eos_token,
        )
        self.vocab_size = len(self.letter_vocab.ch2ix)
        self.max_len = max([max([len(sentence[0]) for sentence in self.sentence_pairs_dataset]), 
                    max([len(sentence[1]) for sentence in self.sentence_pairs_dataset])])
    
