from skimage.io import imread, imshow
from skimage.morphology import binary_closing, disk, skeletonize
from fil_finder import FilFinder2D
import astropy.units as u
import numpy as np
import os
import csv
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

script_dir = os.path.dirname(__file__)

# Path to the main.py directory (one level up from script_dir)
main_dir = os.path.dirname(script_dir)

def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized, scale_percent / 100

def topmost_point(contour):
    return min(contour, key=lambda coord: coord[0][1])[0][1]

def process_image(img):
    original_size = img.shape[:2]
    resized_img, scale_factor = resize_image(img)
    binary_img = (resized_img > 127).astype(np.uint8)
    skeleton = skeletonize(binary_img)
    skeleton_int = (skeleton * 255).astype(np.uint8)

    fil = FilFinder2D(skeleton_int, distance=250 * u.pc, mask=skeleton_int)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

    binary_image = np.array(fil.skeleton_longpath * 255, dtype=np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=topmost_point)
    
    skeleton_counts = [np.sum(cv2.drawContours(np.zeros_like(binary_image), [contour], -1, (255), thickness=1) == 255) / scale_factor for contour in contours]
    return skeleton_counts

def main():
    input_folder = os.path.join(main_dir, 'output','measurement')  # Replace with your images folder path
    output_file = os.path.join(main_dir, 'output','periderm_length.csv')
    
    # First, determine the max number of roots
    max_roots = 0
    for image_name in sorted(os.listdir(input_folder)):
        if image_name.endswith('.png'):
            image_path = os.path.join(input_folder, image_name)
            original_image = imread(image_path, as_gray=True)
            binary_image = original_image > 0.5
            closed_image = binary_closing(binary_image, disk(6))
            counts = process_image((closed_image * 255).astype(np.uint8))
            if len(counts) > max_roots:
                max_roots = len(counts)
    
    # Prepare the column headers based on the max_roots value
    headers = ['Image'] + [f'root_{i+1}' for i in range(max_roots)]

    # Now, write the CSV with the new headers
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the new headers

        for image_name in sorted(os.listdir(input_folder)):
            if image_name.endswith('.png'):
                image_path = os.path.join(input_folder, image_name)
                original_image = imread(image_path, as_gray=True)
                binary_image = original_image > 0.5
                closed_image = binary_closing(binary_image, disk(6))
                counts = process_image((closed_image * 255).astype(np.uint8))

                # Fill in missing lengths with empty strings to match header length
                row_data = [image_name.replace('.png', '')] + counts + [''] * (max_roots - len(counts))
                
                writer.writerow(row_data)

if __name__ == "__main__":
    main()
