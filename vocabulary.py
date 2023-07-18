from collections import Counter, defaultdict
import io
import re


class TokenVocabulary:
    def __init__(self, init_token="<sos>", eos_token="<eos>", pad_token="<pad>", max_tokens=5000):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.ch2ix = defaultdict()
        self.max_tokens = max_tokens
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

        top_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)[:self.max_tokens-4]

        self.ch2ix.update(
            {x: i for i, x in enumerate(Counter(dict(top_tokens)))} # top_tokens is a list
            )

        self.ch2ix[self.init_token] = len(top_tokens)
        self.ch2ix[self.eos_token] = len(top_tokens) + 1
        self.ch2ix[self.pad_token] = len(top_tokens) + 2

        self.ch2ix["<NOT_FOUND>"] = len(top_tokens) + 3  # NOT FOUND TOKEN set to value of vocab_size
        self.vocab_size = len(set(self.ch2ix.values()))
        self.ch2ix.default_factory = (
            lambda: self.ch2ix["<NOT_FOUND>"]
        )  # set default_factory to vocab_size
        self.ix2ch = {v: k for k, v in self.ch2ix.items()}
        self.vocabulary = [self.ix2ch[i] for i in range(self.vocab_size)]

        return self.source_target_list
from collections import Counter, defaultdict
import io
import re


class TokenVocabulary:
    def __init__(self, init_token="<sos>", eos_token="<eos>", pad_token="<pad>", max_tokens=5000):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.ch2ix = defaultdict()
        self.max_tokens = max_tokens
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

        top_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)[:self.max_tokens-4]

        self.ch2ix.update(
            {x: i for i, x in enumerate(Counter(dict(top_tokens)))} # top_tokens is a list
            )

        self.ch2ix[self.init_token] = len(top_tokens)
        self.ch2ix[self.eos_token] = len(top_tokens) + 1
        self.ch2ix[self.pad_token] = len(top_tokens) + 2

        self.ch2ix["<NOT_FOUND>"] = len(top_tokens) + 3  # NOT FOUND TOKEN set to value of vocab_size
        self.vocab_size = len(set(self.ch2ix.values()))
        self.ch2ix.default_factory = (
            lambda: self.ch2ix["<NOT_FOUND>"]
        )  # set default_factory to vocab_size
        self.ix2ch = {v: k for k, v in self.ch2ix.items()}
        self.vocabulary = [self.ix2ch[i] for i in range(self.vocab_size)]

        return self.source_target_list
