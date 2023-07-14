from collections import Counter, defaultdict
import io
import re


class TokenVocabulary:
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
        token_counter = Counter()

        with io.open("train.en", encoding="utf8") as f1, io.open(
            "train.de", encoding="utf8"
        ) as f2:
            for string_source, string_target in zip(f1, f2):
                cleaned_source = " ".join(string_source.split())
                cleaned_target = " ".join(string_target.split())

                self.source_target_list.append(
                    [cleaned_source, cleaned_target]
                )
                for sentence in [cleaned_source, cleaned_target]:
                    tokens = re.findall(r"\w+|[^\s\w]+", sentence)
                    token_counter.update(tokens)

        self.ch2ix.update({x: i for i, x in enumerate(token_counter)})

        self.ch2ix[self.init_token] = len(token_counter)
        self.ch2ix[self.eos_token] = len(token_counter) + 1
        self.ch2ix[self.pad_token] = len(token_counter) + 2

        self.ch2ix["NA"] = len(self.ch2ix)  # NOT FOUND TOKEN set to value of vocab_size
        self.vocab_size = len(self.ch2ix)
        self.ch2ix.default_factory = (
            lambda: self.vocab_size
        )  # set default_factory to vocab_size
        self.ix2ch = {v: k for k, v in self.ch2ix.items()}
        self.vocabulary = [self.ix2ch[i] for i in range(self.vocab_size)]

        return self.source_target_list
