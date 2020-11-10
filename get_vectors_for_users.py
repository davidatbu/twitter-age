from pathlib import Path
import tqdm
import numpy as np
import csv
import os
from embedder import Glove
import typer
from tokenizer import tokenize


def main(input_files_glob: str, output_d: Path) -> None:
    embedder = Glove.from_txt(
        Path(
            "/projectnb/llamagrp/davidat/pretrained_models/glove_twitter/glove.twitter.27B.200d.txt"
            # "glove.twitter.200.200d.txt"
        )
    )

    user_embeddings = []
    users = []
    for user_file in tqdm.tqdm(list(Path('/').glob(input_files_glob)), desc='users processed.'):
        users.append(user_file.stem)
        with user_file.open() as f:
            reader = csv.reader(f)
            tweet_embeddings = []
            headers = next(reader)
            assert headers == [ 'id', 'created_at', 'text']
            for row in reader:
                user_id, date, tweet = row
                tokenized_tweet = tokenize(tweet)
                tweet_embedding = embedder.encode(tokenized_tweet)
                tweet_embeddings.append(tweet_embedding)
            user_embedding = np.stack(tweet_embeddings).mean(axis=0)
            user_embeddings.append(user_embedding)

    np_user_embs = np.stack(user_embeddings)

    with (output_d / "user_embs.npy").open('wb') as fb:
        np.save(fb, np_user_embs)
    with (output_d / "users.txt").open('w') as f:
        f.writelines([u+'\n' for u in users ])



if __name__ == "__main__":
    typer.run(main)
