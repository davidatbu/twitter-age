from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from dnips.iter import myzip
from dnips.nn import pt_pack_unpack
from functools import partial

from more_itertools.more import unzip
import numpy as np

from numpy import ceil

import torch

from torch.tensor import Tensor
import tqdm
from more_itertools import chunked

from logging import FileHandler, Logger, StreamHandler
from dnips.data import JsonlDataset, make_selector
from dnips.nn.tsfmrs import Pipeline


import typer
from pathlib import Path
import pandas as pd
from typing import Iterable, Iterator, List, Sequence, Tuple

logger = Logger(__name__)


def get_last_layers_cls(texts: List[str], pipeline: Pipeline) -> torch.Tensor:
    _, _, hidden_states = pipeline(texts)
    # Take last four layers
    hidden_states = hidden_states[-4:]

    # Flatten the representation.
    flat = torch.cat(hidden_states, dim=-1)
    # Take [CLS] representation
    cls = flat[:, 0, :]

    return cls

def main(
    out_dir: Path,
    users_csv: Path,
    json_dir: Path,
    batch_size: int = 100,
    mdl_name: str = "bert-base-uncased",
) -> None:
    log_file = out_dir / "bert_feature_extract.log"
    logger.addHandler(FileHandler(log_file))
    logger.addHandler(StreamHandler())

    logger.info(
        f"Called with out_dir={out_dir}, users_csv={users_csv}, json_dir={json_dir}, batch_size={batch_size}"
    )

    df_expected_users = pd.read_csv(users_csv, usecols=["Author"], na_filter=False)
    expected_users = set(df_expected_users['Author'].str.strip('@').str.lower())

    # Filter out empty ones
    all_fpaths = list(json_dir.glob("*.jsonl"))
    fpaths = []
    with ThreadPoolExecutor(40) as executor:
        sizes = executor.map(os.path.getsize, all_fpaths)
        for fpath, size in tqdm.tqdm(zip(all_fpaths, sizes), desc='checking file sizes.'):
            if size > 0:
                 fpaths.append(fpath)

    jsonl_dset: Sequence[List[str]] = JsonlDataset(  # type: ignore[assignment]
            fpaths,
        selector=make_selector("renderedContent"),
    )

    pipeline = Pipeline(mdl_name)
    get_last_layers_cls_with_pipeline = partial(get_last_layers_cls, pipeline=pipeline)

    user_feats = []
    users: List[str] = []


    with torch.no_grad():
        pbar = tqdm.tqdm(
            pt_pack_unpack(get_last_layers_cls_with_pipeline, jsonl_dset, batch_size),
            desc=f"users processed.",
            total=len(fpaths),
        )

        for i, emb_per_tweet in enumerate(pbar):
            uname_in_fname = fpaths[i].stem.lower()
            if uname_in_fname not in expected_users:
                logger.error(
                    f"{uname_in_fname} doesn't match any expected users given in CSV."
                )
                continue


            # Concatenate in the batch dim
            pooled = emb_per_tweet.detach().mean(dim=0).cpu()
            user_feats.append(pooled)
            users.append(uname_in_fname)

    cat_user_feats = torch.stack(user_feats, dim=0)
    with (out_dir / "found_usernames.csv").open("w") as f:
        f.writelines(u + "\n" for u in users)

    with (out_dir / "user_tweet_centroids.pth").open("wb") as fb:
        torch.save( cat_user_feats, fb)


if __name__ == "__main__":
    typer.run(main)
