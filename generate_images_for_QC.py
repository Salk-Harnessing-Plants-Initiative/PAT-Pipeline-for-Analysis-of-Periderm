import cv2
import numpy as np
import os

def convert_non_black_to_white(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Could not open or find the image:", image_path)
        return None
    
    # Set non-black pixels to white
    mask_non_black = np.any(image > [0, 0, 0], axis=-1)
    image[mask_non_black] = [255, 255, 255]
    return image

def process_images(image1_path, image2_path, output_path):
    # Load and process image1
    image1 = convert_non_black_to_white(image1_path)
    if image1 is None:
        return

    # Load image2
    image2 = cv2.imread(image2_path)
    if image2 is None:
        print("Could not open or find the image:", image2_path)
        return

    # Convert images to grayscale and then to binary
    _, binary_image1 = cv2.threshold(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

    # Identifying where the white pixels match
    matching_white = cv2.bitwise_and(binary_image1, binary_image2)

    # Identifying where white pixels are exclusive to image1
    exclusive_white_image1 = cv2.bitwise_xor(matching_white, binary_image1)

    # Create a colored version of image1 where the exclusive white pixels are now red
    colored_version = image1.copy()
    colored_version[np.where(exclusive_white_image1 == 255)] = [0, 0, 255]  # BGR format

    # Find the location where the pixels are white in both images
    location_white_both = np.where(matching_white == 255)

    # At these locations, we revert to the original color in image1
    colored_version[location_white_both] = image1[location_white_both]

    # Save the result
    cv2.imwrite(output_path, colored_version)
    print("Processed image saved to:", output_path)

def main():
    folder1 = "/home/lzhang/Desktop/FY_test/Segmentation_upp_v15"
    folder2 = "/home/lzhang/Desktop/FY_test/measurement"
    output_folder = "/home/lzhang/Desktop/FY_test/seg_QC"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder1):
        image_path1 = os.path.join(folder1, filename)
        image_path2 = os.path.join(folder2, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.isfile(image_path2):
            process_images(image_path1, image_path2, output_path)

if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np

def pad_image(image, ref_shape):
    h, w = image.shape[:2]
    ref_h, ref_w = ref_shape[:2]

    padded_image = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)

    h_offset = (ref_h - h) // 2
    w_offset = (ref_w - w) // 2

    padded_image[h_offset:h_offset + h, w_offset:w_offset + w] = image
    return padded_image

def main():
    folder1 = '/home/lzhang/Desktop/FY_test/nature_accession'
    folder2 = '/home/lzhang/Desktop/FY_test/seg_QC'
    output_folder1 = '/home/lzhang/Desktop/FY_test/Ori_pad'
    output_folder2 = '/home/lzhang/Desktop/FY_test/Seg_pad'

    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)

    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)

    for filename in os.listdir(folder1):
        if filename in os.listdir(folder2):
            image1 = cv2.imread(os.path.join(folder1, filename))
            image2 = cv2.imread(os.path.join(folder2, filename))

            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            max_h = max(h1, h2)
            max_w = max(w1, w2)

            padded_image1 = pad_image(image1, (max_h, max_w))
            padded_image2 = pad_image(image2, (max_h, max_w))

            cv2.imwrite(os.path.join(output_folder1, filename), padded_image1)
            cv2.imwrite(os.path.join(output_folder2, filename), padded_image2)

if __name__ == '__main__':
    main()

import cv2
import numpy as np
import os
from pathlib import Path

def find_contours(image_path):
    # Read the image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leftmost_points = []
    for cnt in contours:
        leftmost = tuple(cnt[cnt[:,:,0].argmin()].flatten())
        leftmost_points.append(leftmost)
    # Sort points top to bottom
    leftmost_points.sort(key=lambda x: x[1])
    return leftmost_points

def crop_and_save(image_path, points, save_folder):
    # Read the image
    img = cv2.imread(image_path)
    cropped_images = []
    for i, point in enumerate(points):
        x, y = point
        # Ensure cropping window is within image bounds and has consistent size
        left = max(x - 1000, 0)
        right = min(x + 1600, img.shape[1])
        top = max(y - 400, 0)
        bottom = min(y + 400, img.shape[0])
        
        # Crop the region of interest
        roi = img[top:bottom, left:right]
        
        # Ensure consistent size by padding if necessary
        if roi.shape[0] < 800 or roi.shape[1] < 2600:
            height_pad = max(800 - roi.shape[0], 0)
            width_pad = max(2600 - roi.shape[1], 0)
            roi = cv2.copyMakeBorder(roi, 0, height_pad, 0, width_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
        cropped_images.append(roi)
        # Save the cropped image
        cv2.imwrite(os.path.join(save_folder, f"{Path(image_path).stem}_{i+1}.png"), roi)
    return cropped_images


def stack_and_save(images1, images2, img_name, save_folder):
    for i, (img1, img2) in enumerate(zip(images1, images2)):
        # Stack images vertically
        stacked = np.vstack((img1, img2))
        # Save the stacked image
        cv2.imwrite(os.path.join(save_folder, f"{Path(img_name).stem}_{i+1}.png"), stacked)


def main():
    # Define the paths to your folders
    measurement_folder = '/home/lzhang/Desktop/FY_test/measurement'
    seg_folder = '/home/lzhang/Desktop/FY_test/Seg_pad'
    nature_folder = '/home/lzhang/Desktop/FY_test/Ori_pad'
    output_folder = '/home/lzhang/Desktop/FY_test/for_QC'

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through images in measurement folder
    for img_name in os.listdir(measurement_folder):
        # Ensure we're working with image files only
        if img_name.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            measurement_path = os.path.join(measurement_folder, img_name)
            seg_path = os.path.join(seg_folder, img_name)
            nature_path = os.path.join(nature_folder, img_name)

            # Find leftmost points of contours
            points = find_contours(measurement_path)
            
            # Crop images based on points
            seg_crops = crop_and_save(seg_path, points, seg_folder)
            nature_crops = crop_and_save(nature_path, points, nature_folder)
            
            # Stack cropped images and save
            stack_and_save(nature_crops, seg_crops, img_name, output_folder)  # Added img_name here

if __name__ == "__main__":
    main()
