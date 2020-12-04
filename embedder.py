from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Callable, NamedTuple, overload
import zipfile
from io import TextIOWrapper
from pathlib import Path

from tqdm import tqdm
import numpy as np
import os


class Glove:
    def __init__(self, word_vectors: Dict[str, np.ndarray[np.float64]], dims: int):
        self.word_vectors = word_vectors
        self.dims = dims
        self.oov_vector = np.zeros((self.dims,), dtype=np.float64)

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return [x.lower().strip() for x in text.split()]

    @classmethod
    def from_txt(cls, text_file: Path) -> "Glove":

        # Infer the dims and vocab size
        _, _, vocab_size, dims_str = text_file.stem.split(".")

        dims = int(dims_str.strip("d"))

        with text_file.open() as f_in:
            word_vectors = {}
            for line in tqdm(f_in):
                split_line = line.split()
                word = split_line[0]
                word_vectors[word] = np.array([float(val) for val in split_line[1:]])

        return Glove(word_vectors, dims)

    def encode(self, tokenized_txt: List[str]) -> np.ndarray:
        tok_embs = [
            self.word_vectors.get(token, self.oov_vector) for token in tokenized_txt
        ]
        res = np.stack(tok_embs).mean(axis=0)
        return res
