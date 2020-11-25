import re
import time
from pathlib import Path
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import pandas as pd
import sys


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

DATA_DIR = Path("/projectnb/llamagrp/davidat/twitter_age/")

f = open(DATA_DIR/"vaping.log", "w")
backup = sys.stdout
sys.stdout = Tee(sys.stdout, f)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)

model = BertModel.from_pretrained("bert-base-uncased", config=config)
model.to('cuda')
import numpy as np

df = pd.read_csv(DATA_DIR /"labeledData.csv", encoding="utf-8")
df = df[df["labeled age"].notna()]
adf = pd.DataFrame()
usernames = []
ages = []
centroids = []
data = []
idx = 0
begin = time.time()
for index, row in df.iterrows():
    idx = idx + 1
    username = row["user.name"]
    age = row["labeled age"]
    ndf = pd.read_csv(DATA_DIR / "UserTweets/" / (username + ".csv"), encoding="utf-8")
    ndf = ndf[ndf["text"].notna()]
    ndf["contents"] = ndf["text"].map(lambda x: str(x))
    ndf["contents"] = ndf["contents"].map(lambda x: re.sub('"', "", x))
    ndf["contents"] = ndf["contents"].map(lambda x: re.sub("\n", " ", x))
    ndf["contents"] = ndf["contents"].map(lambda x: x.lower())
    ndf = ndf[ndf["contents"].notna()]
    if ndf.shape[0] == 0:
        continue
    ndf["input_ids"] = ndf["contents"].apply(
        lambda b: torch.tensor(tokenizer.encode(str(b))).unsqueeze(0).cuda()
    )
    ndf["interm"] = ndf["input_ids"].apply(lambda b: model(b))
    ndf["embedding"] = ndf["interm"].apply(
        lambda b: b[0][0][0].tolist()
        + b[2][12][0][0].tolist()
        + b[2][11][0][0].tolist()
        + b[2][10][0][0].tolist()
    )

    # ndf['embedding']=ndf['input_ids'].apply(lambda b: model(b)[0][0][0].tolist())
    print(idx, df.shape[0])
    centroid = np.mean(ndf["embedding"].tolist(), axis=0)
    centroids.append(",".join([str(x) for x in centroid]))
    data.append(centroid)
    usernames.append(username)
    newage = 0
    if age >= 21:
        newage = 1
    ages.append(newage)

total_time = time.time() - begin
print(f"It took {total_time} to process.")
print(np.array(data).shape)
adf["user.name"] = usernames
adf["labeled age"] = ages
adf["centroid"] = centroids
adf.to_csv("tweets_with_embedding_last_four.csv", index=False)

# from keras.utils import np_utils
# from numpy import loadtxt
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.preprocessing import LabelEncoder

sys.stdout = backup
