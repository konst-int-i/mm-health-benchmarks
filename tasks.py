from typing import *
import gdown
import shutil

from mmhb.loader.tcga import encode_patches
from mmhb.utils import detect_os
from mmhb.loader.chestx import preprocess_chestx
from invoke import task
from pathlib import Path
from openslide import OpenSlide
import pandas as pd
import os


@task
def download(
    c,
    dataset: str,
    site: Optional[List[str]] = "brca",
    data_dir: Optional[str] = None,
    samples: Optional[int] = None,
):
    print(site)
    if data_dir is None:
        # set default
        data_dir = Path(f"data/{dataset}/")

    valid_datasets = ["tcga", "mimic", "chestx"]
    assert (
        dataset in valid_datasets
    ), f"Invalid dataset, specify one of {valid_datasets}"

    if dataset == "tcga":
        install_tcga_deps(c)
        download_tcga(c, site, data_dir, samples=samples)

    elif dataset == "mimic":
        download_mimic(c, data_dir, samples=samples)

    elif dataset == "chestx":
        download_chestx(c, data_dir)


def download_mimic(c, data_dir: Path, samples: int = None):
    pass


@task
def download_chestx(c, data_dir: Path):
    """

    Args:
        c:
        data_dir:

    Returns:

    """

    raw_dir = data_dir.joinpath("raw")
    proc_dir = data_dir.joinpath("proc")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading chestx dataset to {raw_dir}...")

    # download PNG images
    if "NLMCXR_png.tgz" not in os.listdir(raw_dir):
        print("Downloading chestx images...")
        c.run(
            f"curl -0 https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz -o {str(raw_dir.joinpath('NLMCXR_png.tgz'))}"
        )
    # download reports
    if "NLMCXR_reports.tgz" not in os.listdir(raw_dir):
        print("Downloading chestx reports...")
        c.run(
            f"curl -0 https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz -o {str(raw_dir.joinpath('NLMCXR_reports.tgz'))}"
        )
    # download term mapping
    if "radiology_vocabulary_final.xlsx" not in os.listdir(raw_dir):
        print("Downloading radiology vocabulary...")
        c.run(
            f"curl -0 https://openi.nlm.nih.gov/imgs/collections/radiology_vocabulary_final.xlsx -o {str(raw_dir.joinpath('radiology_vocabulary_final.xlsx'))}"
        )

    if "TransChex_openi.zip" not in os.listdir(raw_dir):
        print(f"Downloading indeces...")
        id = "1jvT0jVl9mgtWy4cS7LYbF43bQE4mrXAY"
        gdown.download(id=id)
        shutil.move("TransChex_openi.zip", raw_dir.joinpath("TransChex_openi.zip"))

    # unzip
    if "NLMCXR_png" not in os.listdir(raw_dir):
        print("Extracting images...")
        raw_dir.joinpath("NLMCXR_png").mkdir(exist_ok=True)
        c.run(
            f"tar -xvzf {raw_dir.joinpath('NLMCXR_png.tgz')} -C {raw_dir.joinpath('NLMCXR_png')}"
        )
    if "NLMCXR_reports" not in os.listdir(raw_dir):
        print("Extracting reports...")
        raw_dir.joinpath("NLMCXR_reports").mkdir(exist_ok=True)
        c.run(
            f"tar -xvzf {raw_dir.joinpath('NLMCXR_reports.tgz')} -C {raw_dir.joinpath('NLMCXR_reports')}"
        )
    if "TransChex_openi" not in os.listdir(raw_dir):
        print("Extracting indeces...")
        c.run(f"unzip {raw_dir.joinpath('TransChex_openi.zip')} -d {raw_dir}")

    print("ChestX dataset downloaded successfully.")

    print("Preprocessing chestx dataset...")
    preprocess_chestx(raw_dir, proc_dir)


@task
def download_tcga(c, site: str, data_dir: Path, samples: int = None):
    valid_sites = ["brca", "blca", "kirp", "ucec", "hnsc", "paad", "luad", "lusc"]
    # check that all sites are in valid_sites
    assert site in valid_sites, f"Invalid TCGA site, must be one of {valid_sites}"

    # for site in sites:
    print(f"Downloading tcga-{site} dataset...")
    download_dir = data_dir.joinpath(f"wsi/{site}")
    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    manifest_path = Path(f"./data/tcga/gdc_manifests/{site}_wsi_manifest_filtered.txt")

    if samples is not None:
        manifest = pd.read_csv(manifest_path, sep="\t")
        manifest = manifest.sample(n=int(samples), random_state=42)
        tmp_path = manifest_path.parent.joinpath(f"{site}_tmp.txt")
        manifest.to_csv(tmp_path, sep="\t", index=False)
        print(f"Downloading {manifest.shape[0]} files from {site} dataset...")
        c.run(f"./gdc-client download -m {tmp_path} -d {download_dir}")
        # cleanup
        os.remove(tmp_path)
    else:
        command = f"./gdc-client download -m {manifest_path} -d {download_dir}"
        try:
            c.run(command)
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Command: {command}")

    flatten_tcga_dir(c, download_dir)  # flatten directory structure after download
    return None


@task
def install_tcga_deps(c):
    # check if gdc client is installed
    try:
        c.run("./gdc-client --version")
        print(f"gdc-client already installed at {os.getcwd()}")
        return None
    except:
        pass

    system = detect_os()

    print(f"Installing gdc-client for {system}...")
    if system == "linux":
        c.run(
            "curl -0 https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip "
            "--output gdc-client.zip"
        )
        c.run("unzip gdc-client.zip")
    if system == "mac":
        c.run(
            "curl -0 https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_OSX_x64.zip "
            "--output gdc-client.zip"
        )
        c.run("unzip gdc-client.zip")
    elif system == "windows":
        raise NotImplementedError

    print(f"Installed gdc-client at {os.getcwd()}")
    # cleanup
    print(f"Removing zip file")
    os.remove("gdc-client.zip")


@task
def flatten_tcga_dir(c, download_dir: Path):
    """
    Flattens directory structure for WSI images after download using the GDC client from
     `data_dir/*.svs` instead of `data_dir/hash_subdir/*.svs`.
    Args:
        c:
        dataset:
        config:

    Returns:
    """
    # flatten directory structure
    c.run(f"find {download_dir} -type f -name '*.svs' -exec mv {{}} {download_dir} \;")
    c.run(f"find {download_dir} -type f ! -name '*.svs' -exec rm -f {{}} +")


@task
def preprocess(
    c,
    path: str,
    site: str,
    step: str = "patch",
    level: int = 3,
    patch_size: int = 256,
    pretraining: str = "kather",
    check_valid_ids: bool = False,  # rechecks even if valid_id file already exists
):
    wsi_path = Path(path)
    assert os.path.exists(wsi_path), "`wsi_dir` not found"
    assert site in os.listdir(
        wsi_path
    ), "Invalid site, must be directory in specified wsi_dir"
    site_path = wsi_path.joinpath(site)
    prep_path = wsi_path.joinpath(f"{site}_preprocessed_level{level}")
    prep_path.mkdir(parents=True, exist_ok=True)

    valid_steps = ["patch", "features"]
    assert step in valid_steps, f"Invalid step parameter, must be one of {valid_steps}"

    # check which WSIs are available at specified level
    if not os.path.exists(prep_path.joinpath("valid_prep_ids.csv")) or check_valid_ids:
        # check which slides have specified level available (only pass those to preprocessing)
        print(f"Checking available slides at level {level}...")
        valid_ids = []
        for slide_id in os.listdir(site_path):
            # check whether specified level is available in slide
            slide = OpenSlide(site_path.joinpath(f"{slide_id}"))
            try:
                slide.level_dimensions[int(level)]
                valid_ids.append(slide_id)
            except IndexError as e:
                print(f"Level {level} not available for slide {slide_id}")
                continue

            # write temp csv file with valid slide ids to pass to CLAM
            valid_slide_df = pd.DataFrame({"slide_id": valid_ids})
            valid_slide_df.to_csv(prep_path.joinpath("valid_prep_ids.csv"), index=False)

    if step == "patch":
        # generate patches of specified size at given magnification level (3 being lowest magnification)
        if not os.path.exists("CLAM/"):
            c.run("git clone git@github.com:mahmoodlab/CLAM.git")

        c.run(
            f"python CLAM/create_patches_fp.py --source {site_path} --save_dir {prep_path} --process_list valid_prep_ids.csv "
            f"--patch_size {patch_size} --patch_level {int(level)} --seg --patch --stitch"
        )
        # remove CLAM directory
        shutil.rmtree("CLAM")

    if step == "features":
        # only take preprocessed slides
        encode_patches(level, prep_path, site_path, pretraining)
