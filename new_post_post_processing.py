import cv2
import numpy as np

def find_contours_compat(image, mode, method):
    major = cv2.__version__.split(".")[0]
    if major == '3':
        _, contours, _ = cv2.findContours(image.copy(), mode, method)
    else:
        contours, _ = cv2.findContours(image.copy(), mode, method)
    return contours

def process_images():
    image = cv2.imread('Col_gfp.png')
    if image is None:
        print("Error loading image")
        exit()

    threshold = 30
    mask_non_black = np.any(image > threshold, axis=-1)
    white_image = np.ones_like(image) * 255
    final_image = np.where(mask_non_black[..., None], white_image, image)

    gray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours = find_contours_compat(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area_small = 20000
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= max_area_small:
            cv2.drawContours(final_image, [contour], 0, (0, 0, 0), thickness=-1)
    
    return final_image

def post_process_images(final_image):
    image2 = cv2.imread('Col_gfp_1.png')
    if image2 is None:
        print("Could not open or find one of the images.")
        exit()

    _, binary_image1 = cv2.threshold(cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    matching_white = cv2.bitwise_and(binary_image1, binary_image2)
    exclusive_white_image1 = cv2.bitwise_xor(matching_white, binary_image1)
    colored_version = final_image.copy()
    colored_version[np.where(exclusive_white_image1 == 255)] = [0, 0, 255]
    location_white_both = np.where(matching_white == 255)
    colored_version[location_white_both] = final_image[location_white_both]

    return colored_version

def red_to_white(colored_version):
    red_mask = cv2.inRange(colored_version, (0, 0, 255), (0, 0, 255))
    contours = find_contours_compat(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_width, max_width = 30, 1000
    min_height, max_height = 30, 600

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_width <= w <= max_width and min_height <= h <= max_height:
            cv2.drawContours(colored_version, [contour], 0, (255, 255, 255), thickness=-1)

    white_mask = cv2.inRange(colored_version, (255, 255, 255), (255, 255, 255))
    contours = find_contours_compat(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area_small = 6000

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= max_area_small:
            cv2.drawContours(colored_version, [contour], 0, (0, 0, 0), thickness=-1)

    return colored_version

def red_to_black(colored_version):
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([50, 56, 255])
    mask = cv2.inRange(colored_version, lower_red, upper_red)
    colored_version[mask != 0] = [0, 0, 0]

    return colored_version

def keep_right_remove_left(final_image, colored_version):
    original_image_gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    test_image_gray = cv2.cvtColor(colored_version, cv2.COLOR_BGR2GRAY)

    if original_image_gray is None or test_image_gray is None:
        print("Error: Could not load images")
        return

    contours_original = find_contours_compat(original_image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_test = find_contours_compat(test_image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # ... rest of the code remains unchanged

    mask = np.ones(test_image_gray.shape, dtype=np.uint8) * 255

    for contour in contours_original:
        temp_mask = np.zeros(original_image_gray.shape, dtype=np.uint8)
        cv2.drawContours(temp_mask, [contour], -1, (255), thickness=cv2.FILLED)
        matching_contours = []

        for test_contour in contours_test:
            test_mask = np.zeros(test_image_gray.shape, dtype=np.uint8)
            cv2.drawContours(test_mask, [test_contour], -1, (255), thickness=cv2.FILLED)
            intersection = cv2.bitwise_and(temp_mask, test_mask)

            if np.any(intersection):
                matching_contours.append(test_contour)

        if matching_contours:
            rightmost_contour = max(matching_contours, key=lambda c: max(c[:, :, 0]))
            cv2.drawContours(mask, [rightmost_contour], -1, (0), thickness=cv2.FILLED)

    mask = cv2.bitwise_not(mask)
    final_image = cv2.bitwise_and(test_image_gray, mask)
    contours_final = find_contours_compat(final_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_limit = int(final_image.shape[1] * 3 / 5)

    for contour in contours_final:
        x_max = max(contour[:, :, 0])
        if x_max < x_limit:
            cv2.drawContours(final_image, [contour], -1, (0), thickness=cv2.FILLED)

    cv2.imwrite('Col_gfp_final.png', final_image)
    print("Final image saved.")

if __name__ == "__main__":
    final_image = process_images()
    colored_version = post_process_images(final_image)
    colored_version = red_to_white(colored_version)
    colored_version = red_to_black(colored_version)
    keep_right_remove_left(final_image, colored_version)
