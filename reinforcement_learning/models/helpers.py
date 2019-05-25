import numpy as np

def min_max_scale(img):
    img_scaled =  (img - np.max(img)) / (np.max(img) - np.min(img))
    #print(np.max(img), np.min(img))
    #print(np.max(img_scaled), np.min(img_scaled))
    return img_scaled