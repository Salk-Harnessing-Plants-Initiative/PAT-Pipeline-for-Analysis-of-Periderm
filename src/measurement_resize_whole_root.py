from skimage.io import imread, imsave
from skimage.morphology import binary_closing, disk, skeletonize
import numpy as np
import cv2
from fil_finder import FilFinder2D
import astropy.units as u
import time
import os
import csv
from PIL import Image

# Disable PIL image size limit
Image.MAX_IMAGE_PIXELS = None

script_dir = os.path.dirname(__file__)

# Path to the main.py directory (one level up from script_dir)
main_dir = os.path.dirname(script_dir)

def close_image(binary_image):
    return binary_closing(binary_image, disk(6))

def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized, scale_percent / 100

def topmost_point(contour):
    return min(contour, key=lambda coord: coord[0][1])[0][1]

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img, scale_factor = resize_image(img)
    binary_img = (resized_img > 127).astype(np.uint8)
    skeleton = skeletonize(binary_img)
    skeleton_int = (skeleton * 255).astype(np.uint8)
    
    start_time = time.time()
    fil = FilFinder2D(skeleton_int, distance=250 * u.pc, mask=skeleton_int)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
    end_time = time.time()
    
    print(f"Total execution time for {image_path}: {end_time - start_time:.2f} seconds")
    binary_image = np.array(fil.skeleton_longpath * 255, dtype=np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=topmost_point)
    
    skeleton_counts = []
    for contour in contours:
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [contour], -1, (255), thickness=1)
        white_pixels_count = np.sum(mask == 255) / scale_factor
        if white_pixels_count > 2000:
            skeleton_counts.append(white_pixels_count)
    
    return skeleton_counts

def main():
    input_folder = os.path.join(main_dir, 'output','WB_whole_root')  # Source folder for images
    output_file = os.path.join(main_dir, 'output','whole_root_length.csv')  # CSV file for output

    image_skeleton_counts = []
    max_lengths = 0

    # Process each image and find the maximum number of lengths
    image_names = sorted(f for f in os.listdir(input_folder) if f.endswith('.png'))
    for image_name in image_names:
        image_path = os.path.join(input_folder, image_name)
        original_image = imread(image_path, as_gray=True)
        binary_image = original_image > 0.5
        closed_image = close_image(binary_image)

        # Save the closed image temporarily to process it
        temp_closed_image_path = 'temp_closed_image.png'
        imsave(temp_closed_image_path, closed_image.astype(np.uint8) * 255)

        counts = process_image(temp_closed_image_path)
        if len(counts) > max_lengths:
            max_lengths = len(counts)
        image_skeleton_counts.append((image_name.replace('.png', ''), counts))

        # Clean up temporary file
        os.remove(temp_closed_image_path)

    # Create headers with dynamic root names based on max_lengths
    headers = ['Image'] + [f'root_{i+1}' for i in range(max_lengths)]

    # Write the results to the CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for image_name, counts in image_skeleton_counts:
            # Fill in missing lengths with empty strings to match header length
            row = [image_name] + counts + [''] * (max_lengths - len(counts))
            writer.writerow(row)

if __name__ == "__main__":
    main()