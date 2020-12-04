from __future__ import annotations
import numpy as np

import torch
import tqdm
from more_itertools import chunked

from logging import FileHandler, Logger, StreamHandler
from dnips.data import JsonlDataset, make_selector
from dnips.nn.tsfmrs import Pipeline


import typer
from pathlib import Path
import pandas as pd
from typing import List

logger = Logger(__name__)


def main(
    out_dir: Path,
    users_csv: Path,
    json_dir: Path,
    batch_size: int = 500,
    mdl_name: str = "bert-base-uncased",
) -> None:
    log_file = out_dir / "bert_feature_extract.log"
    logger.addHandler(FileHandler(log_file))
    logger.addHandler(StreamHandler())

    logger.info("Called with out_dir, users_csv, json_dir, batch_size")

    pipeline = Pipeline(mdl_name)

    expected_users = set(
        u.strip("@").lower()
        for u in pd.read_csv(
            users_csv,
            encoding="utf-8",
            usecols=[
                "Author",
            ],
            na_filter=False,
        )["Author"].tolist()
    )

    jsonl_dset = JsonlDataset(
        list(json_dir.glob("*.jsonl")),
        selector=make_selector("renderedContent"),
    )

    user_feats = []
    users: List[str] = []

    user_tweets: List[str]
    pbar = tqdm.tqdm(jsonl_dset, desc="users processed")
    with torch.no_grad():
        for uname_in_fname, user_tweets in pbar:  # type: ignore[assignment]
            pbar.set_description(f"{len(users)} users found.")
            uname_in_fname = uname_in_fname.lower()
            if uname_in_fname not in expected_users:
                logger.error(
                    f"{uname_in_fname} doesn't match any expected users given in CSV."
                )
                continue
            if not user_tweets:
                logger.warning(f"Empty file found in {uname_in_fname}.jsonl")
                continue

            feats = []
            for batch in chunked(user_tweets, batch_size):
                _, _, hidden_states = pipeline(batch)
                # Take last four layer
                hidden_states = hidden_states[-4:]

                # Flatten the representation.
                flat = torch.cat(hidden_states, dim=-1)
                # Take [CLS] representation
                cls = flat[:, 0, :]
                feats.append(cls)

            # Concatenate along batch dim
            all_feats = torch.cat(feats, dim=0)
            pooled = all_feats.mean(dim=0).cpu().numpy()
            user_feats.append(pooled)
            users.append(uname_in_fname)

    cat_user_feats = np.stack(user_feats, axis=0)
    with (out_dir / "found_usernames.csv").open("w") as f:
        f.writelines(u + "\n" for u in users)

    with (out_dir / "user_tweet_centroids.npy").open("wb") as fb:
        np.save(fb, cat_user_feats)


if __name__ == "__main__":
    typer.run(main)

############## TESTS
