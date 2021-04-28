import os
import re
from collections import Counter

import torch
from torch.utils.data import TensorDataset
from torchtext.vocab import Vocab

from datasets import load_dataset

from universal_transformer import DATA_DIR_PATH
from universal_transformer.class_registry import registry, register_class


@register_class(("dataset", "babi"))
class BabiDataset:
    def __init__(self, tokenizer, debug=False):
        self.debug = debug
        train_path = os.path.join(DATA_DIR_PATH, "babi", "en-valid", "qa1_train.txt")
        self.train = self.path_to_dataset(train_path, tokenizer, fit_tokenizer=True)

        val_path = os.path.join(DATA_DIR_PATH, "babi", "en-valid", "qa1_valid.txt")
        self.val = self.path_to_dataset(val_path, tokenizer)

    def path_to_dataset(self, path, tokenizer, fit_tokenizer=False):
        stories, answers = list(zip(*self.read_babi_lines(path)))
        if self.debug:
            stories = stories[:10]
            answers = answers[:10]
        if fit_tokenizer:
            tokenizer.fit(stories + answers)
        stories, story_attn_masks = texts_to_tensors(stories, tokenizer)
        answers, answer_attn_masks = texts_to_tensors(answers, tokenizer)
        return TensorDataset(stories, answers, story_attn_masks, answer_attn_masks)

    @classmethod
    def read_babi_lines(cls, path):
        story_lines = []
        with open(path) as f:
            for line in f:
                line_num = int(line.split(" ")[0])
                line = cls.clean_line(line)
                if line_num == 1 and story_lines:
                    story_lines = []
                if "?" in line:
                    question, answer = re.split(r"(?<=\?)\s*", line)
                    story = " ".join(story_lines + [question])
                    yield story, answer
                else:
                    story_lines.append(line)

    @staticmethod
    def clean_line(line):
        line = re.sub(r"^[\d\s]*", "", line, flags=re.MULTILINE)
        line = re.sub(r"[\d\s]*$", "", line, flags=re.MULTILINE)
        return line


@register_class(("dataset", "lambada"))
class LambadaDataset:
    def __init__(self, tokenizer=None, debug=False):
        self.debug = debug

        path = os.path.join(DATA_DIR_PATH, "lambada", "lambada-vocab-2.txt")
        with open(path) as f:
            lines = [line.strip().split('\t') for line in f]
            wordCounts = Counter({line[0]: int(line[1]) for line in lines})

        self.vocab = Vocab(wordCounts)

        data = load_dataset('lambada') 
        train = data['train']
        train.remove_columns_(['domain'])
        val = data['validation']
        val.remove_columns_(['domain'])
        test = data['test']
        test.remove_columns_(['domain'])

        self.train = (texts_to_tensors_lambada(train, self.vocab, True), )
        self.val = (texts_to_tensors_lambada(val, self.vocab, False), )
        self.test = (texts_to_tensors_lambada(test, self.vocab, False), )
        
        print('data loaded')



def texts_to_tensors(texts, tokenizer):
    """Convert a sequence of texts and labels to a dataset."""
    token_ids_seqs = tokenizer.encode_texts(texts)
    seq_length_max = len(token_ids_seqs[0])

    pad_token_id = tokenizer.token_to_id[tokenizer.pad_token]
    lengths = [
        ids.index(pad_token_id) if ids[-1] == pad_token_id else len(ids)
        for ids in token_ids_seqs
    ]
    att_masks = [[1] * length + [0] * (seq_length_max - length) for length in lengths]

    token_ids_seqs = torch.tensor(token_ids_seqs, dtype=torch.long)
    att_masks = torch.tensor(att_masks, dtype=torch.bool)

    return token_ids_seqs, att_masks


def texts_to_tensors_lambada(texts, vocab, split=False):

    vec = []

    if not split:
        for instance in texts:
            tensor = torch.tensor([vocab[token] for token in instance['text'].split()])
            tensor = torch.cat([tensor, torch.ones(203 - len(tensor))])
            #print(tensor.shape)
            vec.append(tensor.unsqueeze(0))

    else:
        for instance in texts:
            words = instance['text'].split()
            start = 0
            for i in range(203, len(words), 203):
                tensor = torch.tensor([vocab[token] for token in words[start:i]])
                tensor = torch.cat([tensor, torch.ones(203 - len(tensor))]) # add padding
                #print(tensor.shape)
                vec.append(tensor.unsqueeze(0))
                start = i

    vec2D = torch.cat(vec, axis=0)
    print(vec2D.shape)
    return vec2D 

def get_dataset(config, tokenizer=None):
    key = ("dataset", config.dataset)
    if key in registry:
        cls, kwargs = registry[key]
        accepted_args = set(cls.__init__.__code__.co_varnames)
        accepted_args.remove("self")
        kwargs.update(
            {k.replace("dataset.", ""): v for k, v in config.items() if "dataset." in k}
        )
        kwargs["tokenizer"] = tokenizer
        return cls(**kwargs)

    raise KeyError("Dataset not found!")
