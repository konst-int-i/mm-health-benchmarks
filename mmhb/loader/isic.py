import multiprocessing
import torch
import torchvision.models as models
from torchvision.transforms.functional import to_tensor
from PIL import Image
from pathlib import Path
import pandas as pd
import os


def preprocess_isic(n: int = None):
    isic_path = Path("/auto/archive/tcga/other_data/ISIC/")
    df = pd.read_csv(isic_path.joinpath("train.csv"), index_col="patient_id")
    img_index = df["image_name"]
    raw_path = isic_path.joinpath("jpeg/train/")
    if n is None:
        filenames = (img_index.iloc[:] + ".jpg").values
    else:
        filenames = (img_index.iloc[:16] + ".jpg").values

    with multiprocessing.Pool() as pool:
        pool.map(_path_img, [(raw_path, filename) for filename in filenames])


def _path_img(args) -> torch.Tensor:
    """
    Patches image and saves the preprocesed
    Args:
        raw_path:
        filename:
        patch_dims:

    Returns:

    """
    raw_path, filename = args
    patch_dims = 3
    write_path = Path("../mm-lego/data/isic/img_prep", mkdir=True)
    overwrite = False
    save_file = filename.replace(".jpg", ".pt")
    if not overwrite:
        if save_file in os.listdir(write_path):
            print(f"File {save_file} already exists in {write_path}")
            return None

    img_path = raw_path.joinpath(filename)

    # Load a pre-trained ResNet model
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    # Remove the last layer (classification layer)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Function to divide an image into patches
    def get_patches(image, patch_size=224, stride=224):
        patches = []
        for i in range(0, image.width, stride):
            for j in range(0, image.height, stride):
                patch = image.crop((i, j, i + patch_size, j + patch_size))
                patches.append(patch)
        return patches

    # Load the image
    image = Image.open(img_path)

    # resize image to nxn patches
    image = image.resize((int(1.5 * patch_dims * 224), patch_dims * 224))

    # Get patches
    patches = get_patches(image)

    # Encode each patch
    encoded_patches = []  # treat as tensor
    for i, patch in enumerate(patches):
        # Convert the patch to a tensor and add a batch dimension
        patch_tensor = to_tensor(patch).unsqueeze(0)
        patch_tensor = patch_tensor.to(device)

        # Pass the patch through the model
        encoded_patch = model(patch_tensor)
        # Remove the batch dimension and add the encoded patch to the list
        encoded_patches.append(encoded_patch.detach().squeeze())
    # convert to tensor
    encoded_patches = torch.stack(encoded_patches)

    if write_path is not None:
        # save_file = filename.replace(".jpg", ".pt")
        # write_path = Path(write_path, mkdir=True)
        torch.save(encoded_patches, write_path.joinpath(save_file))
        print(f"Saved encoded patches to {write_path.joinpath(save_file)}")

    print(f"Written {len(os.listdir(write_path))}/{len(os.listdir(raw_path))} files")

    return encoded_patches


if __name__ == "__main__":
    preprocess_isic()
    # isic_path = Path("/auto/archive/tcga/other_data/ISIC/")
    # df = pd.read_csv(isic_path.joinpath("train.csv"), index_col="patient_id")
    # img_index = df["image_name"]
    # raw_path = isic_path.joinpath("jpeg/train/")
    # # filename = f"{img_index.iloc[0]}.jpg"
    # filenames = (img_index.iloc[:16] + ".jpg").values

    # map multiprocessing
    # map multiprocessing
    # with multiprocessing.Pool() as pool:
    #     pool.map(_path_img, [(raw_path, filename) for filename in filenames])
    # with multiprocessing.Pool() as pool:
    #     pool.map(patch_img, [raw_path] * len(filenames), filenames)

    # print(len(filenames))
    # patch_img(raw_path, filename, patch_dims=3)
    # img_path = isic_path.joinpath("jpeg/train").joinpath(f"{img_index.iloc[0]}.jpg")

    # Load an image
    # img_path = Path("data/ISIC_0000000.jpg")
    # encoded_patches = preprocess(img_path)
    # print(encoded_patches.shape)
    # print(encoded_patches)
