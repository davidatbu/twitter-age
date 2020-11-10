import typer
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def main(input_d: Path) -> None:

    with open(input_d / "users.txt") as f:
        users_ls = [l.strip() for l in f]

    with open(input_d / "user_embs.npy", "rb") as fb:
        user_embs = np.load(fb)

    sorting_indices, users_ls = map(list, zip(*sorted(enumerate(users_ls), key=lambda pair: pair[1])))  # type: ignore[assignment]
    user_embs = user_embs[sorting_indices]

    user_ages_df = pd.read_csv(str(input_d / "user_ages.csv"), na_filter=False)
    assert list(user_ages_df.columns) == ["user.name", "labeled age"]

    user_ages_df.sort_values("user.name", inplace=True) 
    assert (user_ages_df['user.name'] == users_ls).all() 
    user_ages = user_ages_df['labeled age']

    having_age = (user_ages != "")
    if (~having_age).any():
        print(f"{(~having_age).sum()} users have no labelled age. Will use the other {having_age.sum()} users.")
    user_ages = user_ages[having_age]
    user_embs = user_embs[having_age.to_numpy()]
    users = np.array(users_ls)[having_age.to_numpy()]

    random_indices = np.array(list(range(len(users))))
    np.random.shuffle(random_indices)

    print("Shuffing ..")
    users = users[random_indices]
    user_embs = user_embs[random_indices]
    user_ages = user_ages.iloc[random_indices]

    user_ages = user_ages.astype('float') >= 21

    # ipdb> user_ages.value_counts()
    # True     969
    # False    557
    # Name: labeled age, dtype: int64


    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300000))
    scoring = ["precision_macro", "recall_macro", "accuracy"]
    scores = cross_validate(clf, user_embs, user_ages.to_numpy(), cv=5, scoring=scoring)
    for score in scoring:
        fold_averaged = scores['test_'  + score ].mean()
        print(f'{score}: {fold_averaged}')


if __name__ == "__main__":
    typer.run(main)
