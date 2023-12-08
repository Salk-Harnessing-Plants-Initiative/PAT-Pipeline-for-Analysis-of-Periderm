import cv2
import numpy as np
from skimage.morphology import skeletonize
from fil_finder import FilFinder2D
import astropy.units as u
import time
import os
import csv

# Read data from the CSV file
# Get the directory of the script file
script_dir = os.path.dirname(__file__)

# Path to the main.py directory (one level up from script_dir)
main_dir = os.path.dirname(script_dir)

def topmost_point(contour):
    return min(contour, key=lambda coord: coord[0][1])[0][1]

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary_img = (img > 127).astype(np.uint8)
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
    for i, contour in enumerate(contours, start=1):
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [contour], -1, (255), thickness=1)
        white_pixels_count = np.sum(mask == 255)
        #print(f"skeleton {i}: {white_pixels_count}")
        skeleton_counts.append(white_pixels_count)
    
    return skeleton_counts

def main():
    input_folder = os.path.join(main_dir, 'output','measurement')
    output_file = os.path.join(main_dir, 'output','periderm_length.csv')
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'root_1', 'root_2', 'root_3', 'root_4', 'root_5', 'root_6', 'root_7'])  # Adjust as needed
        
        for image_name in os.listdir(input_folder):
            if image_name.endswith(".png"):
                image_path = os.path.join(input_folder, image_name)
                counts = process_image(image_path)
                writer.writerow([image_name.replace('.png', '')] + counts)

if __name__ == "__main__":
    #Suppres all warnings
    #warnings.filterwarnings("ignore")
    main()
