import numpy as np
import cv2
from scipy.stats import entropy
from scipy.ndimage import center_of_mass

#Feature 1 : measures how dark or bright a digit image is.
def total_intensity(image):

    return np.sum(image) / (image.size * 16) 

#Feature 2 : measures how pixel intensity is distributed from top to bottom.
def top_middle_bottom_ratio(image):
    top = image[:3, :]
    middle = image[3:5, :]
    bottom = image[5:, :]
    top_sum = np.sum(top)
    middle_sum = np.sum(middle)
    bottom_sum = np.sum(bottom)
    total = top_sum + middle_sum + bottom_sum + 1e-6  

    return top_sum / total, middle_sum / total, bottom_sum / total

#Feature 3 : measures how much the left and right halves look alike.
def horizontal_symmetry(image):
    left = image[:, :4]
    right = np.fliplr(image[:, 4:])

    return np.mean(left - right)

#Feature 4 : tells the position of the digit with respect to the center of mass.
def center_of_mass_x(image):
    _, x = center_of_mass(image)

    return x

#Feature 5 : measures the transition pattern of the pixel from 0 to 1 or vice versa.
def transition_count(image):
    binary = (image > 0).astype(int)
    row_transitions = np.sum(np.abs(np.diff(binary, axis=1)))
    col_transitions = np.sum(np.abs(np.diff(binary, axis=0)))

    return row_transitions + col_transitions

#Feature 6 : measures the complexity of the digit.
def pixel_entropy(image):
    hist, _ = np.histogram(image, bins=16, range=(0, 16), density=True)

    return entropy(hist + 1e-8)

#Feature 7 : measure the coverage area of the digit in the image.
def ink_spread(image):
    binary = image > 0
    coords = np.argwhere(binary)
    if coords.size == 0:

        return 0.0
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    bounding_box_area = (max_row - min_row + 1) * (max_col - min_col + 1)
    ink_pixels = np.sum(binary)
    
    return ink_pixels / bounding_box_area if bounding_box_area > 0 else 0.0

#Feature 8 : measures the distribution of intensities in the regions.
def quadrant_sums(image):
    h, w = image.shape
    half_h, half_w = h // 2, w // 2
    q1 = np.sum(image[:half_h, :half_w])
    q2 = np.sum(image[:half_h, half_w:])
    q4 = np.sum(image[half_h:, half_w:])
    return q1, q2, q4

#Feature 9 : measures the curvature and contours in the digit.
def contour_features(image):
    image_uint8 = (image * 255).astype(np.uint8)
    _, binary = cv2.threshold(image_uint8, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    avg_length = np.mean([cv2.arcLength(cnt, True) for cnt in contours]) if contours else 0.0
    
    return num_contours, avg_length

def extract_features(X):
    features = []
    for x in X:
        image = x.reshape(8, 8)
        top, middle, bottom = top_middle_bottom_ratio(image)
        q1, q2, q4 = quadrant_sums(image)
        num_contours, avg_perimeter = contour_features(image)
        row = [
            total_intensity(image),
            top, middle, bottom,
            horizontal_symmetry(image),
            center_of_mass_x(image),
            transition_count(image),
            pixel_entropy(image),
             ink_spread(image),           
            q1, q2, q4,             
            num_contours, avg_perimeter,
        ]
        features.append(row)
    return np.array(features)

