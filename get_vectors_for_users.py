from pathlib import Path
from embeddings import GloveEmbedding, FastTextEmbedding, KazumaCharEmbedding, ConcatEmbedding
import typer
from tokenizer import tokenize

def main(input_files_glob: str, output_d) -> None:
    embedder = GloveEmbedding(file_name='/home/davidat/Downloads/glove.twitter.27B.50d.txt.zip', d_emb=50)



if __name__ == '__main__':
    typer.run(main)

