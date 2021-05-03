import torch
import torch.nn as nn

from universal_transformer.class_registry import registry, register_class
from universal_transformer.transformers import (
    UniversalTransformer,
    VanillaTransformer,
)


@register_class(("model", "vanilla_transformer"), transformer_class=VanillaTransformer)
@register_class(
    ("model", "universal_transformer"), transformer_class=UniversalTransformer
)
class TransformerModelBase(nn.Module):
    def __init__(self, embedding_matrix, transformer_class=None, vocab_size=None, embedding_size=None, **kwargs):
        super().__init__()
        self.transformer_class = transformer_class
        if embedding_matrix is not None:
            self.embedding_size = embedding_matrix.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_matrix)
            )
            self.vocab_size = embedding_matrix.shape[0]

        else:
            self.embedding_size = embedding_size
            self.vocab_size = vocab_size
            #self.embedding_size = kwargs['kwargs']['embedding_size']
            #self.vocab_size = kwargs['kwargs']['vocab_size']
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.transformer = self.transformer_class(
            d_model=self.embedding_size, **kwargs
        )
        self.output_linear = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(
        self,
        *,
        source_ids,
        target_ids,
        source_padding_mask=None,
        target_padding_mask=None
    ):
        source_ids = self.embedding(source_ids)
        target_ids = self.embedding(target_ids)

        source_ids = source_ids.permute(1, 0, 2)
        target_ids = target_ids.permute(1, 0, 2)

        print(target_ids.size(0))
        att_mask = self.transformer.generate_square_subsequent_mask(
            target_ids.size(0)
        )
        att_mask = att_mask.to(source_ids.device)

        output = self.transformer(
            src=source_ids,
            tgt=target_ids,
            src_mask=att_mask,
            tgt_mask=att_mask,
            memory_mask=att_mask,
            src_key_padding_mask=source_padding_mask,  # 1 means ignore.
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        output = output.permute(1, 0, 2)
        output = self.output_linear(output)
        return output


def get_model(config, embedding_matrix=None, vocab=None, embedding_size=None):
    key = ("model", config.model)
    if key in registry:
        cls, kwargs = registry[key]
        accepted_args = set(cls.__init__.__code__.co_varnames)
        accepted_args.remove("self")
        kwargs.update(
            {k.replace("model.", ""): v for k, v in config.items() if "model." in k}
        )
        if embedding_matrix is not None:
            kwargs["embedding_matrix"] = embedding_matrix
        else:
            kwargs2 = {}
            print('alt kwargs')
            kwargs["embedding_matrix"] = None
            kwargs["vocab_size"] = len(vocab.itos)
            kwargs["embedding_size"] = embedding_size 
            #kwargs2["vocab_size"] = len(vocab.itos)
            #kwargs2["embedding_size"] = embedding_size 
            #kwargs['kwargs'] = kwargs2


        return cls(**kwargs)

    raise KeyError("Model not found!")
