import numpy as np

def total_intensity(image):
    return np.sum(image)

def top_bottom_ratio(image):
    top = np.sum(image[:4 , :])
    bottom = np.sum(image[4: , :])
    return top / (bottom + 0.000001)

def horizontal_symmetry(image):
    left = image[: , :4]
    right = np.fliplr(image[: , 4:])
    return -np.sum(np.abs(left - right))

def extract_features(X):
    features = []
    for x in X:
        image = x.reshape(8,8)
        row = [ total_intensity(image), top_bottom_ratio(image), horizontal_symmetry(image)]
        features.append(row)
    return np.array(features)
    

