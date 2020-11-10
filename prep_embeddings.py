import zipfile
import tqdm
import numpy as np
from typing import Dict
from io import  TextIOWrapper
import typer
from pathlib import Path

    
def main(zip_: Path, out_d: Path=Path('.'), dim_wanted: str='200d'):

    z = zipfile.ZipFile(zip_)
    files_in_zip = (z.infolist())

    dim_wanted = '.' + dim_wanted + '.'

    matching_files = [ f for f in files_in_zip if dim_wanted in f.filename ]


    if len(matching_files) != 1:
        raise Exception(f"File passed contains {len(matching_files)} files matching the dim specification '{dim_wanted}'.")

    file_wanted = matching_files[0] 

    word_to_id: Dict[str, int] = {}
    all_vecs = []
    with z.open(file_wanted) as fb:
        for line in tqdm.tqdm(TextIOWrapper(fb)):
            parts = line.strip().split()
            word= parts[0]
            word_to_id[word] = len(word_to_id)
            vec = [ float(i) for i in parts[1:] ]
            all_vecs.append(vec)

    np_vec = np.array(all_vecs)

    with open(out_d.joinpath("words.txt", 'w')) as f:
        f.writelines(list(word + '\n' for word in word_to_id))

    with open((out_d / "vectors.npy"), 'wb') as fb:
        np.save(fb, np_vec)




if __name__ == "__main__":
    typer.run(main)
