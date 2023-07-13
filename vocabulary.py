from collections import Counter, defaultdict
import io


class LetterVocabulary:
    def __init__(self, init_token="<sos>", eos_token="<eos>", pad_token="<pad>"):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.ch2ix = defaultdict()
        self.ix2ch = {}
        self.vocab_size = None
        self.vocabulary = None
        self.source_target_list = []

    def build_vocabulary(self, source_file="train.en", target_file="train.de"):
        letter_counts = Counter()
        source_target_list = []

        with io.open("train.en", encoding="utf8") as f1, io.open(
            "train.de", encoding="utf8"
        ) as f2:
            for string_en, string_de in zip(f1, f2):
                self.source_target_list.append(
                    [" ".join(string_en.split()), " ".join(string_de.split())]
                )

        # Iterate over the pairs and count the letters individually
        for pair in self.source_target_list:
            for letter in " ".join(pair):
                letter_counts[letter] += 1
        self.ch2ix.update({x[0]: i for i, x in enumerate(letter_counts)})

        self.ch2ix[self.init_token] = len(letter_counts)
        self.ch2ix[self.eos_token] = len(letter_counts) + 1
        self.ch2ix[self.pad_token] = len(letter_counts) + 2

        self.vocab_size = len(self.ch2ix)
        self.ch2ix["NA"] = self.vocab_size  # NOT FOUND TOKEN set to value of vocab_size
        self.ch2ix.default_factory = (
            lambda: self.vocab_size
        )  # set default_factory to vocab_size
        self.ix2ch = {v: k for k, v in self.ch2ix.items()}
        self.vocabulary = [self.ix2ch[i] for i in range(self.vocab_size + 1)]

        return source_target_list
