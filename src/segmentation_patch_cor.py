# -*- coding: utf-8 -*-
# %%
from datetime import datetime

start = datetime.now()

import os, cv2, math, csv
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album

import segmentation_models_pytorch.utils

#! pip install opencv-python
#!pip install -q -U segmentation-models-pytorch albumentations > /dev/null
import segmentation_models_pytorch as smp

script_dir = os.path.dirname(__file__)

# Path to the main.py directory (one level up from script_dir)
main_dir = os.path.dirname(script_dir)


# add pading and crop images to patch size
def add_0padding_crop(
    patch_size,
    overlap_size,
    image_path,
    image_path_padding,
    image_path_crop,
    pad_image=False,
):
    """Add zero padding to the size of image and crop it for patch.

    Args:
        patch_size: expected patch size of deep learning model.
        overlap_size: the expected overlap/border of two adjacent images.
        image_path: the path where store original images to be padded.
        image_path_padding: the expected path to save padding images.
        image_path_crop: the expected path to save cropped images.
        pad_image: boolean data, where True means return padding images.

    Returns
        Add padding of current image and save padding images.
    """
    color = [0, 0, 0]  # add zero padding

    image_name = [
        file
        for file in os.listdir(image_path)
        if (file.endswith(".png") and not file.startswith("."))
    ]

    for name in image_name:
        # pass
        im = cv2.imread(image_path + name)
        shape_0, shape_1 = im.shape[0], im.shape[1]
        n_0, n_1 = math.ceil(shape_0 / (patch_size - overlap_size / 2)), math.ceil(
            shape_1 / (patch_size - overlap_size / 2)
        )
        top, bottom = math.ceil(
            (n_0 * (patch_size - overlap_size / 2) - shape_0) / 2
        ), math.floor((n_0 * (patch_size - overlap_size / 2) - shape_0) / 2)
        left, right = math.ceil(
            (n_1 * (patch_size - overlap_size / 2) - shape_1) / 2
        ), math.floor((n_1 * (patch_size - overlap_size / 2) - shape_1) / 2)
        im_pad = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        if pad_image:
            name_pad = image_path_padding + name
            cv2.imwrite(name_pad, im_pad)

        idx = 0
        for i in range(n_0):
            for j in range(n_1):
                idx += 1
                crop_name = str(os.path.splitext(name)[0]) + "_" + str(idx) + ".png"
                top = i * (patch_size - overlap_size)
                left = j * (patch_size - overlap_size)
                im_crop = im_pad[top : top + patch_size, left : left + patch_size, :]
                name_crop = image_path_crop + crop_name
                cv2.imwrite(name_crop, im_crop)


# %%
# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace("_", " ").title(), fontsize=20)
        plt.imshow(image)
    plt.show()


# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x


# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


# %%
class BackgroundDataset(torch.utils.data.Dataset):
    """Stanford Background Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
        self,
        df,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.image_paths = df["image_path"].tolist()
        self.mask_paths = df["label_colored_path"].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        # print(self.image_paths[i])
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)


# for testing dataset, similar as BackgroundDataset, but will not only return image, mask, but also return name
class BuildingsDataset(torch.utils.data.Dataset):
    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
        self,
        df,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
    ):

        self.image_paths = df["image_path"].tolist()
        self.mask_paths = df["label_colored_path"].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        # print(self.image_paths)
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        names = self.image_paths[i].rsplit("/", 1)[-1].split(".")[0]
        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask, names

    def __len__(self):
        # return length of
        return len(self.image_paths)


# %%
def get_training_augmentation():
    train_transform = [
        # album.PadIfNeeded(min_height=550, min_width=660, always_apply=True, border_mode=0),
        # LW height and width change from 832 to 1000 to 1984 to 2720
        album.RandomCrop(height=patch_size, width=patch_size, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.5,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        # LW size should be square and can be devided by 32
        # LW height and width change from 992 to 1120 to 2752
        album.PadIfNeeded(
            min_height=patch_size,
            min_width=patch_size,
            always_apply=True,
            border_mode=0,
            value=(0, 0, 0),
        ),
    ]
    return album.Compose(test_transform)


def get_test_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        # LW change min_height of 1120 to 6016 to 2016, same as val
        # LW change min_width of 992 to 4000 to 1120 for checking images instead of images_test
        album.PadIfNeeded(
            min_height=patch_size,
            min_width=patch_size,
            always_apply=True,
            border_mode=0,
            value=(0, 0, 0),
        ),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


# %%
# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.join(
    main_dir,
    "models",
    "best_model_unet_plusplus_resnet34_periderm4classes_Lin_0301_1024_6.pth",
)
alt_model_path = (
    "../input/unet-with-pretrained-resnet50-encoder-pytorch/best_model_eye_ce.pth"
)

# Check if the primary model file exists
if os.path.exists(model_path):
    best_model = torch.load(model_path, map_location=DEVICE)
    print("Loaded UNet model from this run.")

# If not, check for an alternative model file
elif os.path.exists(alt_model_path):
    best_model = torch.load(alt_model_path, map_location=DEVICE)
    print("Loaded UNet model from a previous commit.")

# If neither model file is found
else:
    print("No model found.")


# %%
# Center crop padded image / mask to original image dims
def crop_image(image, true_dimensions):
    return album.CenterCrop(p=1, height=true_dimensions[0], width=true_dimensions[1])(
        image=image
    )


sample_preds_folder = os.path.join(main_dir, "output", "Prediction_patch")
if not os.path.exists(sample_preds_folder):
    os.makedirs(sample_preds_folder)

files = os.listdir(sample_preds_folder)
if len(files) > 0:
    for items in files:
        os.remove(os.path.join(sample_preds_folder, items))


# %%
# zero padding and crop

patch_size = 1024
overlap_size = 16

# image_path = "./Images_all/"
image_path = os.path.join(main_dir, "nature_accession/")
print(image_path)

# flip left and right
# Output folder (optional, or you can overwrite)
output_path = os.path.join(main_dir, "nature_accession_flip\\")
os.makedirs(output_path, exist_ok=True)

for filename in os.listdir(image_path):
    img = cv2.imread(os.path.join(image_path, filename))
    flipped_img = cv2.flip(img, 1)  # 1 means horizontal flip
    cv2.imwrite(os.path.join(output_path, filename), flipped_img)

image_path = output_path

image_path_padding = os.path.join(main_dir, "output", "Image_Padding")
if not os.path.exists(image_path_padding):
    os.mkdir(image_path_padding)

# image_path_padding = "./Plate_Image_Padding_temp/" if os.path.exists("./Plate_Image_Padding_temp/") else os.mkdir("./Plate_Image_Padding_temp/")
files_temp = os.listdir(image_path_padding)
if len(files_temp) > 0:
    for items in files_temp:
        os.remove(os.path.join(image_path_padding, items))

image_path_crop = os.path.join(main_dir, "output", "Image_Crop/")
if not os.path.exists(image_path_crop):
    os.mkdir(image_path_crop)


# image_path_crop = "./Plate_Image_Crop_temp/" if os.path.exists("./Plate_Image_Crop_temp/") else os.mkdir("./Plate_Image_Crop_temp/")
files_temp = os.listdir(image_path_crop)
if len(files_temp) > 0:
    for items in files_temp:
        os.remove(os.path.join(image_path_crop, items))

pad_image = False

add_0padding_crop(
    patch_size, overlap_size, image_path, image_path_padding, image_path_crop, pad_image
)
image_name = sorted(os.listdir(image_path))
print("Finish padding ", len(image_name), "images.", flush=True)


# %%
# %% generate metadata file

# image_path = image_path_crop

subimage_list = [
    file
    for file in os.listdir(image_path_crop)
    if (file.endswith(".png") and not file.startswith("."))
]

metadata_row = []
for i in range(len(subimage_list)):
    image_path_i = image_path_crop + subimage_list[i]
    label_path_i = image_path_crop + subimage_list[i]
    metadata_row.append([str(i + 1), image_path_i, label_path_i])

metadata_file = os.path.join(main_dir, "models", "metadata_tem.csv")

header = ["image_id", "image_path", "label_colored_path"]
with open(metadata_file, "w") as csvfile:
    writer = csv.writer(csvfile, lineterminator="\n")
    writer.writerow([g for g in header])
    for x in range(len(metadata_row)):
        writer.writerow(metadata_row[x])


# %%
class PredictionDataset(torch.utils.data.Dataset):
    """Stanford Background Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
        self,
        df,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.image_paths = df["image_path"].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        # print(self.image_paths[i])
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        names = self.image_paths[i].rsplit("/", 1)[-1].split(".")[0]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        return image, names

    def __len__(self):
        # return length of
        return len(self.image_paths)


# %%

preds_folder = image_path_crop

DATA_DIR = main_dir
pred_df = pd.read_csv(os.path.join(main_dir, "models", "metadata_tem.csv"))

select_classes = ["background", "periderm", "endodermis", "lateral_root"]

class_dict = pd.read_csv(os.path.join(DATA_DIR, "models", "label_class_dict_lr.csv"))
# Get class names
class_names = class_dict["name"].tolist()
# Get class RGB values
class_rgb_values = class_dict[["r", "g", "b"]].values.tolist()

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

# select_class_rgb_values =  np.array([[0, 0, 0], [255, 255, 255]])
ENCODER = "resnet34"
# ENCODER = 'resnet101'
# LW can choose different architecture, use the resnet101 to test
ENCODER_WEIGHTS = "imagenet"
CLASSES = select_classes
ACTIVATION = (
    "sigmoid"  # could be None for logits or 'softmax2d' for multiclass segmentation
)
# ACTIVATION = 'softmax2d'
# create segmentation model with pretrained encoder
model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
# model = nn.DataParallel(model, device_ids=[0]) # without cuda
model = nn.DataParallel(model, device_ids=[0]).cuda()
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

test_dataset = PredictionDataset(
    pred_df,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

sample_preds_folder = os.path.join(main_dir, "output", "Segmentation_temp/")
if not os.path.exists(sample_preds_folder):
    os.mkdir(sample_preds_folder)

# sample_preds_folder = "./Plate_segmentation_temp/" if os.path.exists("./Plate_segmentation_temp/") else os.mkdir("./Plate_segmentation_temp/")
files = os.listdir(sample_preds_folder)
if len(files) > 0:
    for items in files:
        os.remove(os.path.join(sample_preds_folder, items))


test_dataset_vis = BackgroundDataset(
    pred_df,
    class_rgb_values=select_class_rgb_values,
)


# %%
for idx in range(len(test_dataset)):  # len(test_dataset)

    # print(sample_preds_folder)
    image, names = test_dataset[idx]
    image_vis = test_dataset_vis[idx][0].astype("uint8")
    true_dimensions = image_vis.shape
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    # Predict test image
    pred_mask = best_model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    # Convert pred_mask from `CHW` format to `HWC` format
    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    # Get prediction channel corresponding to foreground
    pred_foreground_heatmap = crop_image(
        pred_mask[:, :, select_classes.index("periderm")], true_dimensions
    )["image"]
    pred_mask = crop_image(
        colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values),
        true_dimensions,
    )["image"]
    # Convert gt_mask from `CHW` format to `HWC` format
    # LW gt_mask = np.transpose(gt_mask,(1,2,0))
    # LW gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values), true_dimensions)['image']
    # cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])
    # cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), pred_foreground_heatmap)
    cv2.imwrite(os.path.join(sample_preds_folder, f"{names}.png"), pred_mask)
    # visualize(
    #    original_image = image_vis,
    # LW ground_truth_mask = gt_mask,
    #    predicted_mask = pred_mask,
    #    pred_foreground_heatmap = pred_foreground_heatmap
    # )

# %%

print("Finish segmenting", flush=True)


def stitch_crop_images(
    patch_size, overlap_size, original_image_path, image_path, stitch_path
):
    """Stitch the prediction in patch size.

    Args:
        patch_size: expected patch size of deep learning model.
        overlap_size: the expected overlap/border of two adjacent images.
        original_image_path: the path where store original images.
        image_path: the path where store prediction in patch size.
        stitch_path: the expected path where store the stitched predictions.


    Returns:
        Save stitched predictions.

    """
    PredNameList = [
        file
        for file in os.listdir(image_path)
        if (file.endswith(".png") and not file.startswith("."))
    ]

    image_name = []
    for i in range(len(PredNameList)):
        filename = os.path.splitext(PredNameList[i])[0]
        filename2 = filename.rsplit("_", 1)
        image_name.append(filename2[0])
    image_name = np.array(image_name)
    image_name_unique = np.unique(image_name)

    for name in image_name_unique:
        # pass
        image_crop_name_list = [
            file
            for file in os.listdir(image_path)
            if (
                file.endswith(".png")
                and file.startswith(str(name))
                and not file.startswith(".")
            )
        ]
        image_crop_count = len(image_crop_name_list)

        # will need the original image shape
        im = cv2.imread(original_image_path + name + ".png")
        shape_0, shape_1, shape_2 = im.shape[0], im.shape[1], im.shape[2]
        n_0, n_1 = math.ceil(shape_0 / (patch_size - overlap_size / 2)), math.ceil(
            shape_1 / (patch_size - overlap_size / 2)
        )

        n_0_idx = []
        n_1_idx = []
        index = []
        ind = 0
        # index of row and column
        for i in range(n_0):
            for j in range(n_1):
                ind += 1
                n_0_idx.append(i)
                n_1_idx.append(j)
                index.append(ind)

        ind_df = pd.DataFrame({"n_0_idx": n_0_idx, "n_1_idx": n_1_idx, "index": index})
        ind_array = np.array(ind_df)

        # ind = 0
        im_stitch = np.zeros(
            [
                int(n_0 * (patch_size - overlap_size / 2)),
                int(n_1 * (patch_size - overlap_size / 2)),
                shape_2,
            ]
        )
        for i in range(n_0):
            for j in range(n_1):
                # ind += 1
                top = i * (patch_size - overlap_size)
                left = j * (patch_size - overlap_size)
                ind = np.squeeze(
                    ind_array[
                        np.where((ind_array[:, 0] == i) & (ind_array[:, 1] == j)), 2
                    ]
                )
                im_pred_patch = cv2.imread(image_path + name + "_" + str(ind) + ".png")
                # im_crop = im_pad[top:top+patch_size, left:left+patch_size, :]
                if top == 0 and left == 0:
                    im_stitch[top : top + patch_size, left : left + patch_size, :] = (
                        im_pred_patch
                    )
                elif top == 0 and left > 0:
                    left_ind = np.squeeze(
                        ind_array[
                            np.where(
                                (ind_array[:, 0] == i) & (ind_array[:, 1] == j - 1)
                            ),
                            2,
                        ]
                    )

                    im_left = cv2.imread(
                        image_path + name + "_" + str(left_ind) + ".png"
                    )

                    # get the overlap area (leaft side of im_pred_patch, right side of im_pred_patch_left)
                    im_pred_patch_left = im_pred_patch[0:patch_size, 0:overlap_size, :]
                    im_left_right = im_left[0:patch_size, -overlap_size:, :]

                    # calculate maximum value of overlapping area
                    im_stitch[top : top + patch_size, left : left + overlap_size, :] = (
                        np.maximum(im_pred_patch_left, im_left_right)
                    )
                    im_stitch[
                        top : top + patch_size,
                        left + overlap_size - 1 : left + patch_size - 1,
                        :,
                    ] = im_pred_patch[
                        0:patch_size, overlap_size - 1 : patch_size - 1, :
                    ]

                elif top > 0 and left == 0:
                    top_ind = np.squeeze(
                        ind_array[
                            np.where(
                                (ind_array[:, 0] == i - 1) & (ind_array[:, 1] == j)
                            ),
                            2,
                        ]
                    )

                    im_top = cv2.imread(image_path + name + "_" + str(top_ind) + ".png")

                    # get the overlap area (top side of im_pred_patch, bottom side of im_pred_patch_top)
                    im_pred_patch_top = im_pred_patch[0:overlap_size, 0:patch_size, :]
                    im_top_bottom = im_top[-overlap_size:, 0:patch_size, :]

                    # calculate maximum value of overlapping area
                    im_stitch[top : top + overlap_size, left : left + patch_size, :] = (
                        np.maximum(im_pred_patch_top, im_top_bottom)
                    )
                    im_stitch[
                        top + overlap_size - 1 : top + patch_size - 1,
                        left : left + patch_size,
                        :,
                    ] = im_pred_patch[
                        overlap_size - 1 : patch_size - 1, 0:patch_size, :
                    ]
                else:
                    top_ind = np.squeeze(
                        ind_array[
                            np.where(
                                (ind_array[:, 0] == i - 1) & (ind_array[:, 1] == j)
                            ),
                            2,
                        ]
                    )
                    left_ind = np.squeeze(
                        ind_array[
                            np.where(
                                (ind_array[:, 0] == i) & (ind_array[:, 1] == j - 1)
                            ),
                            2,
                        ]
                    )

                    im_top = cv2.imread(image_path + name + "_" + str(top_ind) + ".png")
                    # get the overlap area (top side of im_pred_patch, bottom side of im_pred_patch_top)
                    im_pred_patch_top = im_pred_patch[0:overlap_size, 0:patch_size, :]
                    im_top_bottom = im_top[-overlap_size:, 0:patch_size, :]

                    im_stitch[top : top + overlap_size, left : left + patch_size, :] = (
                        np.maximum(im_pred_patch_top, im_top_bottom)
                    )
                    im_stitch[
                        top + overlap_size - 1 : top + patch_size - 1,
                        left : left + patch_size,
                        :,
                    ] = im_pred_patch[
                        overlap_size - 1 : patch_size - 1, 0:patch_size, :
                    ]

                    im_left = cv2.imread(
                        image_path + name + "_" + str(left_ind) + ".png"
                    )
                    # get the overlap area (leaft side of im_pred_patch, right side of im_pred_patch_left)
                    im_pred_patch_left = im_pred_patch[0:patch_size, 0:overlap_size, :]
                    im_left_right = im_left[0:patch_size, -overlap_size:, :]

                    # calculate maximum value of overlapping area
                    im_stitch[top : top + patch_size, left : left + overlap_size, :] = (
                        np.maximum(im_pred_patch_left, im_left_right)
                    )
                    im_stitch[
                        top : top + patch_size,
                        left + overlap_size - 1 : left + patch_size - 1,
                        :,
                    ] = im_pred_patch[
                        0:patch_size, overlap_size - 1 : patch_size - 1, :
                    ]

        name_stitch = stitch_path + name + ".png"
        cv2.imwrite(name_stitch, im_stitch)


# %%
# original_image_path = "./Images_all/"
original_image_path = image_path
image_path = os.path.join(main_dir, "output", "Segmentation_temp/")
stitch_path = os.path.join(DATA_DIR, "output", "Segmentation_upp_v15/")
if not os.path.exists(stitch_path):
    os.makedirs(stitch_path)

stitch_crop_images(
    patch_size, overlap_size, original_image_path, image_path, stitch_path
)

print("Finish stitching", flush=True)

# %% postposing


def get_average(matrix):
    # use 0 as x axis, 1 as y axis to keep consist with image
    center_x = np.mean(matrix[:, 0])
    center_y = np.mean(matrix[:, 1])
    return center_x, center_y


def closest_value(array, input_value):
    index = (np.abs(array - input_value)).argmin()
    return index


def buffer(bbox, buffer_ratio_x, buffer_ratio_y):
    """Ruturn the bounding box with a buffer area.

    Args:
        bbox: tuple of bounding box of (left, top, width, height)
        buffer_ratio_x: the ratio of buffer width to the original bounding box width.
        buffer_ratio_y: the ratio of buffer height to the original bounding box height.

    Return:
        Tuple of the buffered bounding box (left, top, width, height)
    """
    left, top, width, height = bbox
    center_x = left + int(width / 2)
    center_y = top + int(height / 2)
    new_left = center_x - int(width * buffer_ratio_x / 2)
    new_top = center_y - int(height * buffer_ratio_y / 2)
    if new_left < 0:
        new_left = 0
    if new_top < 0:
        new_top = 0
    return new_left, new_top, int(width * buffer_ratio_x), int(height * buffer_ratio_y)


image_path = os.path.join(main_dir, "output", "Segmentation_upp_v15/")
save_path = os.path.join(main_dir, "output", "Post_processing_v09/")
if not os.path.exists(save_path):
    os.makedirs(save_path)


imageList = [file for file in os.listdir(image_path) if file.endswith(".png")]
imageList.sort()

for i in range(len(imageList)):
    image_name = os.path.join(image_path, imageList[i])
    image = cv2.imread(image_name)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(gray, 89, 255, cv2.THRESH_BINARY)
    # plt.imshow(binary)

    # remove small connected area
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
        binary
    )

    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1

    min_size_small = 100  # 200 # 300
    im_result = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        bbox2 = stats[blob + 1, 0:4]
        if sizes[blob] >= min_size_small:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 100

    im_result = im_result.astype(np.uint8)
    ret, binary = cv2.threshold(im_result, 50, 255, cv2.THRESH_BINARY)
    # plt.imshow(binary)

    inverted_binary = ~binary
    # contours, hierarchy = cv2.findContours(inverted_binary,
    #  cv2.RETR_TREE,
    #  cv2.CHAIN_APPROX_SIMPLE)
    # findContours returns 2 values in OpenCV 3 and 3 values in OpenCV 4+
    contours, hierarchy = cv2.findContours(
        inverted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2:]

    contours_filter = []
    for j in range(len(contours)):
        contour = contours[j]
        area = cv2.contourArea(contour)
        if area > 1000 and area < (image.shape[0] * image.shape[1] / 2):
            contours_filter.append(contours[j])

    # sort filtered contour
    contours_filter_list = []
    for j in range(len(contours_filter)):
        center_x, center_y = get_average(
            np.squeeze(contours_filter[j])
        )  # the most accurate way is tosort based on a range of y instead of exact y value of center point
        new_cenxy = center_y * 1000 - center_x / 1000
        contours_filter_list.append([j, center_x, center_y, new_cenxy])
    contours_filter_list = list(contours_filter_list)
    contours_filter_list = sorted(
        contours_filter_list, key=lambda item: item[3], reverse=False
    )
    contours_filter_index = np.array(contours_filter_list)

    contours_filter_sort = []
    for j in range(len(contours_filter)):
        index = int(contours_filter_index[j, 0])
        contouri = contours_filter[index]
        contours_filter_sort.append(contouri)

    # plt.imshow(np.squeeze(contours_filter_sort[0]))
    # cv2.contourArea(contours_filter_sort[0])

    buffer_ratio_x = 1.5
    buffer_ratio_y = 3

    image = cv2.imread(image_name)
    segmentation_revert = image
    for k in range(len(contours_filter_sort)):
        for j in range(k, len(contours_filter_sort)):
            contour1, contour2 = [], []
            contour1, contour2 = np.squeeze(contours_filter_sort[k]), np.squeeze(
                contours_filter_sort[j]
            )

            # contour1, contour2 = np.squeeze(contours[0][4:]), np.squeeze(contours[1][4:])

            contour1_x, contour1_y = get_average(contour1)
            contour2_x, contour2_y = get_average(contour2)

            if not (contour1_x < contour2_x):
                contour1, contour2 = contour2, contour1
                contour1_x, contour1_y = get_average(contour1)
                contour2_x, contour2_y = get_average(contour2)

            if contour1_x < contour2_x:  # contour1 at left
                # get the right x coordination of the left one and left x of the right one
                right1_x = np.percentile(contour1[:, 0], 99)  # contour1
                left2_x = np.percentile(contour2[:, 0], 1)  # contour2

                # get the y axis of the right one at very left
                contour2_y = contour2[closest_value(contour2[:, 0], int(left2_x)), 1]

                contour1_y = contour1[closest_value(contour1[:, 0], int(right1_x)), 1]

                contour1_area = cv2.contourArea(contour1)
                contour2_area = cv2.contourArea(contour2)

                if (
                    (left2_x - right1_x) < 150
                    and (left2_x - right1_x) >= 0
                    and abs(contour2_y - contour1_y) < 150
                ):  # original is 150, 50
                    bbox = [
                        int(right1_x),
                        np.min([contour2_y, contour1_y]),
                        int(abs(right1_x - left2_x)),
                        int(abs(contour2_y - contour1_y)),
                    ]
                    bbox_buffer = buffer(bbox, buffer_ratio_x, buffer_ratio_y)
                    # image = cv2.imread(image_name)[1000:2000:,:]
                    # segmentation_revert = image
                    segmentation_bbox = segmentation_revert[
                        bbox_buffer[1] : bbox_buffer[1] + bbox_buffer[3],
                        bbox_buffer[0] : bbox_buffer[0] + bbox_buffer[2],
                        :,
                    ]
                    segmentation_bbox[
                        (
                            (segmentation_bbox[:, :, 0] == 128)
                            & (segmentation_bbox[:, :, 1] == 0)
                            & (segmentation_bbox[:, :, 2] == 0)
                        )
                    ] = [128, 128, 0]
                    segmentation_bbox[
                        (
                            (segmentation_bbox[:, :, 0] == 0)
                            & (segmentation_bbox[:, :, 1] == 128)
                            & (segmentation_bbox[:, :, 2] == 0)
                        )
                    ] = [128, 128, 0]
                    segmentation_revert[
                        bbox_buffer[1] : bbox_buffer[1] + bbox_buffer[3],
                        bbox_buffer[0] : bbox_buffer[0] + bbox_buffer[2],
                        :,
                    ] = segmentation_bbox
                    # cv2.imwrite("Y:/Lin_Wang/FY_training/image_revert.png",segmentation_revert)
                elif (left2_x - right1_x) > 149 and abs(
                    contour2_y - contour1_y
                ) < 1000:  # 1000 # and contour1_area<contour2_area/2
                    # bbox = [int(right1_x), np.min([contour2_y,contour1_y]), int(abs(right1_x- left2_x)), int(abs(contour2_y-contour1_y))]
                    left, top = np.percentile(contour1[:, 0], 1), np.percentile(
                        contour1[:, 1], 1
                    )
                    right, bottom = np.percentile(contour1[:, 0], 99), np.percentile(
                        contour1[:, 1], 99
                    )
                    width, height = right - left, bottom - top
                    bbox = int(left), int(top), int(width), int(height)
                    bbox_buffer = buffer(bbox, buffer_ratio_x, buffer_ratio_y)
                    # image = cv2.imread(image_name)[1000:2000:,:]
                    # segmentation_revert = image
                    segmentation_bbox = segmentation_revert[
                        bbox_buffer[1] : bbox_buffer[1] + bbox_buffer[3],
                        bbox_buffer[0] : bbox_buffer[0] + bbox_buffer[2],
                        :,
                    ]
                    segmentation_bbox[
                        (
                            (segmentation_bbox[:, :, 0] == 128)
                            & (segmentation_bbox[:, :, 1] == 128)
                            & (segmentation_bbox[:, :, 2] == 0)
                        )
                    ] = [
                        128,
                        0,
                        0,
                    ]  # [128,0,0]
                    # np.max(segmentation_bbox[2])
                    segmentation_revert[
                        bbox_buffer[1] : bbox_buffer[1] + bbox_buffer[3],
                        bbox_buffer[0] : bbox_buffer[0] + bbox_buffer[2],
                        :,
                    ] = segmentation_bbox
                    # cv2.imwrite("Y:/Lin_Wang/FY_training/image_revert.png",segmentation_revert)
                else:
                    bbox = [
                        int(right1_x),
                        np.min([contour2_y, contour1_y]),
                        int(abs(right1_x - left2_x)),
                        int(abs(contour2_y - contour1_y)),
                    ]
                    bbox_buffer = buffer(bbox, buffer_ratio_x, buffer_ratio_y)
                    # image = cv2.imread(image_name)[1000:2000:,:]
                    # segmentation_revert = image
                    segmentation_bbox = segmentation_revert[
                        bbox_buffer[1] : bbox_buffer[1] + bbox_buffer[3],
                        bbox_buffer[0] : bbox_buffer[0] + bbox_buffer[2],
                        :,
                    ]
                    # segmentation_bbox[((segmentation_bbox[:,:,0] == 128) & (segmentation_bbox[:,:,1] == 128) & (segmentation_bbox[:,:,2] == 0))] = [128,0,0]
                    segmentation_revert[
                        bbox_buffer[1] : bbox_buffer[1] + bbox_buffer[3],
                        bbox_buffer[0] : bbox_buffer[0] + bbox_buffer[2],
                        :,
                    ] = segmentation_bbox

    new_name = os.path.join(save_path, imageList[i])
    cv2.imwrite(new_name, segmentation_revert)

print("Finish post-processing", flush=True)
# %% convert periderm
image_path = os.path.join(main_dir, "output", "Post_processing_v09/")
save_path = os.path.join(main_dir, "output", "segmentation_upp_periderm_v04/")

if not os.path.exists(save_path):
    os.mkdir(save_path)

# sample_preds_folder = "./Plate_segmentation_temp/" if os.path.exists("./Plate_segmentation_temp/") else os.mkdir("./Plate_segmentation_temp/")
files = os.listdir(save_path)
if len(files) > 0:
    for items in files:
        os.remove(os.path.join(save_path, items))


imageList = [file for file in os.listdir(image_path) if file.endswith(".png")]
imageList.sort()

for i in range(len(imageList)):
    image_name = os.path.join(image_path, imageList[i])
    image = cv2.imread(image_name)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(gray, 89, 255, cv2.THRESH_BINARY)

    # remove small connected area
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
        binary
    )

    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1

    min_size_small = 30000  # 200
    im_result = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        bbox2 = stats[blob + 1, 0:4]
        if sizes[blob] >= min_size_small:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 100

    im_result = im_result.astype(np.uint8)
    ret, binary = cv2.threshold(im_result, 50, 255, cv2.THRESH_BINARY)

    new_name = save_path + imageList[i]
    cv2.imwrite(new_name, binary)

print("Finish converting", flush=True)
# %% closing
image_path = os.path.join(main_dir, "output", "segmentation_upp_periderm_v04/")
save_path_1 = os.path.join(main_dir, "output", "Final_Periderm_Segmentation_Results/")

if not os.path.exists(save_path_1):
    os.mkdir(save_path_1)

# sample_preds_folder = "./Plate_segmentation_temp/" if os.path.exists("./Plate_segmentation_temp/") else os.mkdir("./Plate_segmentation_temp/")
# files = os.listdir(save_path)
# if len(files)>0:
#    for items in files:
#        os.remove(os.path.join(save_path,items))


imageList = [file for file in os.listdir(image_path) if file.endswith(".png")]
imageList.sort()

for i in range(len(imageList)):
    image_name = os.path.join(image_path, imageList[i])
    image = cv2.imread(image_name)
    kernel = np.ones((20, 10), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image_save_name = os.path.join(save_path_1, imageList[i])
    cv2.imwrite(image_save_name, closing)

difference = datetime.now() - start
print(difference)

print("Finish!", flush=True)
