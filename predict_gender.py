from io import BytesIO
from itertools import takewhile
from typing import Counter
from functools import cached_property
import threading
from more_itertools import bucket
import csv
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from PIL import Image  # type: ignore
import urllib
from urllib.error import URLError, HTTPError
from simdjson import Parser, dumps as json_dumps  # type: ignore
import requests
from m3inference import get_lang
from m3inference.m3inference import M3Inference
import typer
import tqdm
from pathlib import Path
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    Executor,
    as_completed,
)
import requests  # to get image from the web
import shutil  # to save it locally

app = typer.Typer()

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


def get_extension(img_url: str) -> str:
    if "." not in img_url.split("/")[-1]:
        return "png"
    dotpos = img_url.rfind(".")
    extension = img_url[dotpos + 1 :]
    if extension.lower() == "gif":
        return "png"
    return extension


def resize_img(
    img_f: BytesIO,
    img_out_path: Path,
    filter=Image.BILINEAR,
    force: bool = False,
    url: str = None,
) -> bool:
    try:
        img = Image.open(img_f).convert("RGB")
        if img.size[0] + img.size[1] < 400 and not force:
            logger.info(f"{url} is too small. Skip.")
            return False
        img = img.resize((224, 224), filter)
        img.save(img_out_path)
        return True
    except Exception as e:
        logger.warning(
            f"Couldn't resize {url}. The error is {e}. "
            f"The attmpted filename is {str(img_out_path)}"
        )
        return False


def download_resize_img(
    url: str, file_stem: str, out_dir: Path, skip_if_exists: bool
) -> Optional[Path]:
    ext = get_extension(url)
    img_fpath = out_dir / f"{file_stem}.{ext}"
    if img_fpath.exists():
        return img_fpath
    try:
        img_data = urllib.request.urlopen(url)
        img_data = img_data.read()
    except (HTTPError, ValueError) as err:
        logger.warning(
            "Couldn't fetch profile image from Twitter."
            f" URL: {url} . "
            f"Error was: {err}"
        )
        return None

    if resize_img(BytesIO(img_data), img_fpath, force=True, url=url):
        return img_fpath
    else:
        return None


thread_local = threading.local()


_T = TypeVar("_T")


def try_until(
    func: Callable[..., _T], *args: Any, **kwargs: Any
) -> Callable[[Iterable[Any]], _T]:
    """Make a function that tries abunch of values until it succeeds in getting a truthy
    val."""
    not_set = object()

    def trier(items: Iterable[Any]) -> _T:
        val = not_set
        for item in items:
            if val := func(item, *args, **kwargs):
                return val
        if val is not_set:
            raise StopIteration(
                f"An empty iterator ({items}) was passed to try_until()"
            )
        return val  # type: ignore[return-value]

    return trier


def prep_one_jsonl(fpath: Path, img_output_dir: Path) -> Optional[Any]:
    if not hasattr(thread_local, "parser"):
        thread_local.parser = Parser()
    parser = thread_local.parser

    img_urls: Set[str] = set()
    screen_name = (
        fpath.stem.lower()
    )  # From filenaem because snscrape fetches old usernames in metadata sometimes
    with fpath.open() as f:
        first_line = f.readline().strip().encode()
        if not first_line:  # File empty
            return None
        doc = parser.parse(first_line)
        # Read these from the first tweet
        id_ = str(doc.at_pointer("/user/id"))
        name = doc.at_pointer("/user/displayname")
        description = doc.at_pointer("/user/description")
        img_urls.add(
            doc.at_pointer("/user/profileImageUrl").replace("_normal", "_400x400")
        )

        # Read all profile image urls to try them all
        for line in f:
            doc = parser.parse(line.strip().encode())
            img_urls.add(
                doc.at_pointer("/user/profileImageUrl").replace("_normal", "_400x400")
            )

    ready_json = {
        "id": id_,
        "name": name,
        "screen_name": screen_name,
        "description": description,
    }

    if len(img_urls) > 1:
        print(f"Trying {len(img_urls)} urls for {screen_name}")
    retrier = try_until(
        download_resize_img,
        file_stem=screen_name,
        out_dir=img_output_dir,
        skip_if_exists=True,
    )
    if not (img_fpath := retrier(img_urls)):
        logger.error(f"Downloading image failed for: {ready_json['screen_name']}")
        ready_json["img_path"] = None
    else:
        ready_json["img_path"] = str(img_fpath.resolve())
    ready_json["lang"] = get_lang(ready_json["description"])

    return ready_json


@app.command()
def prepare_for_m3(
    jsonl_dir: Path, output_dir: Path, num_workers: int = 4, do_threads: bool = True
) -> None:
    executor_class: Union[Type[ThreadPoolExecutor], Type[ProcessPoolExecutor]]
    if do_threads:
        executor_class = ThreadPoolExecutor
    else:
        executor_class = ProcessPoolExecutor
    futures = []
    with executor_class(num_workers) as executor:
        pbar = tqdm.tqdm(list(jsonl_dir.glob("*.jsonl")), desc="Distributed users")
        for jsonl_fpath in pbar:
            future = executor.submit(
                prep_one_jsonl,
                fpath=jsonl_fpath,
                img_output_dir=output_dir / "img",
            )
            futures.append(future)

        with open(output_dir / "data.jsonl", "w") as f:
            futures_pbar = tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="writing results"
            )
            text_successful = 0
            img_successful = 0
            for future in futures_pbar:
                res = future.result()
                futures_pbar.set_description(
                    f"Images found for {img_successful}. Text found for: {text_successful}."
                )
                if not res:
                    continue
                text_successful += 1
                if res["img_path"]:
                    img_successful += 1
                j = json_dumps(res) + "\n"
                f.write(j)


@app.command()
def do_predict(jsonl_fpath: Path, out_csv_fpath: Path, batch_size:int=650) -> None:
    # With images
    import json

    json_has_img = lambda j: j["img_path"] is not None
    with jsonl_fpath.open() as f:
        lines = map(json.loads, map(lambda x: x.strip(), f))
        b = bucket(lines, json_has_img)
        all_ = []
        for has_img in b:
            jsons = list(b[has_img])
            jsons = jsons
            print(
                f"Goint to predict on batches that "
                + ("has" if has_img else "doesn't have")
                + " images."
            )
            m3 = M3Inference(use_full_model=has_img)
            results = m3.infer(jsons, batch_size=batch_size)
            del m3

            # See if we had some duplicate ids in the input data
            if len(jsons) != list(results):
                cntr = Counter(j['id'] for j in jsons)
                dups = list(takewhile(lambda i: i[1]>1, cntr.most_common()))
                print(f"The following are dup ids and counts: {dups}")


            done = set()
            for j in jsons:
                r = results[j["id"]]
                if j["id"] in done:
                    continue
                done.add(j["id"])
                all_.append((j["screen_name"], r["gender"]["male"], r["org"]["is-org"]))

    headers = ("screen_name", "probability_of_being_male", "probability_of_being_org")
    all_ = [headers] + all_
    with out_csv_fpath.open("w") as f:
        writer = csv.writer(f)
        writer.writerows(all_)


if __name__ == "__main__":
    app()
