import json
import random
import logging

import torchaudio
import pandas as pd
import pathlib as pl

from tqdm import tqdm

from speechbrain.utils.logger import get_logger
from speechbrain.dataio.dataio import load_pkl, save_pkl

OPT_FILE = "opt_covost2_prepare.pkl"

TGT_METADATA = {
    "train": "train.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
}

SRC_AUDIO = "clips"

log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = get_logger(__name__)

def prepare_covost2(
    save_folder,
    src_audio_folder,
    tgt_translation_folder,
    splits=["train", "dev", "test"],
    src_lang="de",
    tgt_lang="en",
    seed=1234,
    skip_prep=False
    ):

    config = {
        "src_audio_folder": src_audio_folder,
        "tgt_translation_folder": tgt_translation_folder,
        "splits": splits,
        "save_folder": save_folder,
        "seed": seed,
    }

    if skip_prep and skip(splits, save_folder, config):
        logger.info("Data preparation is already done. Skipping...")
        return

    random.seed(seed)
    TGT_Trans_METADATA = f"covost_v2.{src_lang}_{tgt_lang}"

    save_folder = pl.Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    src_audio = pl.Path(src_audio_folder) / SRC_AUDIO

    tgt_train = pl.Path(tgt_translation_folder) / f"{TGT_Trans_METADATA}.train.tsv"
    tgt_dev = pl.Path(tgt_translation_folder) / f"{TGT_Trans_METADATA}.dev.tsv"
    tgt_test = pl.Path(tgt_translation_folder) / f"{TGT_Trans_METADATA}.test.tsv"

    save_opt = pl.Path(save_folder) / OPT_FILE
    save_json_train = pl.Path(save_folder) / "train.json"
    save_json_dev = pl.Path(save_folder) / "dev.json"
    save_json_test = pl.Path(save_folder) / "test.json"

    msg = "\tCreating json file for CoVost2 Dataset.."
    logger.info(msg)

    if "train" in splits:
      prepare_json(
          save_json_train,
          src_audio,
          tgt_train,
      )

    if "dev" in splits:
      prepare_json(
          save_json_dev,
          src_audio,
          tgt_dev,
      )

    if "test" in splits:
      prepare_json(
          save_json_test,
          src_audio,
          tgt_test,
      )

    save_pkl(config, save_opt)


def prepare_json(json_file, src_audio_folder, tgt_translation_path):
    src_audio_folder = pl.Path(src_audio_folder)
    tgt_translation_path = pl.Path(tgt_translation_path)
    json_file = pl.Path(json_file)

    tgt_translation_df = pd.read_csv(tgt_translation_path, sep="\t")

    json_dict = {}

    logger.info(f"Processing {len(tgt_translation_df)} audio files for {json_file}...")

    for _, row in tqdm(tgt_translation_df.iterrows(), total=len(tgt_translation_df), desc="Processing audio files"):
        audio_filename = row["path"]
        tgt_text = row["translation"]

        src_audio_path = src_audio_folder / audio_filename
        if not src_audio_path.is_file():
            continue

        src_sig, sr = torchaudio.load(str(src_audio_path))
        duration = src_sig.shape[1] / sr

        json_dict[audio_filename] = {
            "src_audio": str(src_audio_path),
            "tgt_text": tgt_text,
            "duration": duration,
        }

    json_file.write_text(json.dumps(json_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"{json_file} successfully created!")

def skip(splits, save_folder, conf):

    save_folder = pl.Path(save_folder)
    split_files = {
        "train": "train.json",
        "dev": "dev.json",
        "test": "test.json",
    }

    skip = True
    for split in splits:
        if not (save_folder / split_files[split]).is_file():
            skip = False

    save_opt = save_folder / OPT_FILE
    if skip:
        if save_opt.is_file():
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                return True
            else:
                return False
        else:
            return False
    return skip


if __name__ == "__main__":
    prepare_covost2(
        save_folder = "/content/drive/MyDrive/Colab Notebooks/Model-Merging/covost2_audio/covost2_prepared",
        src_audio_folder = "/content/drive/MyDrive/Colab Notebooks/Model-Merging/covost2_audio/cv-corpus-20.0-de/cv-corpus-20.0-2024-12-06/de",
        tgt_translation_folder = "/content/drive/MyDrive/Colab Notebooks/Model-Merging/covost_translation_extracted",
        splits=["train", "dev", "test"],
        src_lang="de", tgt_lang="en", seed=1234, skip_prep=False,
    )
