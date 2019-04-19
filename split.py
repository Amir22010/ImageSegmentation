# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:17:07 2019

@author: Amir.Khan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils.image import read_image, draw_bbox
from utils.data import get_relevant_frames, get_boxes

labels = pd.read_csv("pool_labels.csv")
#labels = pd.read_csv("car_labels.csv")


paths = labels[["Frame", "Mask"]].as_matrix()
paths = paths[:, 0] + "!" + paths[:, 1]
paths = np.unique(paths)

def temp(img_path: str, mask_path: str) -> None:
    """Open IMAGE and MASK and plot"""
    img = read_image(img_path)
    mask = read_image(mask_path, gray=True)

    result = cv2.bitwise_and(img, img, mask=mask)
    plt.imshow(result)
    
# Select 10 random examples
idx_list = np.random.randint(low = 0, high = paths.shape[0], size=5 * 2)

plt.figure(1, figsize=(50, 50))
for i, idx in enumerate(idx_list):
    plt.subplot(5, 2, i + 1)
    temp(*paths[idx].split("!"))
    
    
def plot_bbox(image_path: str, df: pd.DataFrame) -> None:
    image = read_image(image_path)
    df = get_relevant_frames(image_path, df)
    boxes = get_boxes(df)
    
    for box in boxes:
        image = draw_bbox(image, box.left_top, box.right_bot, color=(0, 255, 0), thickness=3)
    
    plt.imshow(image)
    
    
# Select 10 random examples
idx_list = np.random.randint(low = 0, high = paths.shape[0], size=5 * 2)

plt.figure(1, figsize=(50, 50))
for i, idx in enumerate(idx_list):
    plt.subplot(5, 2, i + 1)
    plot_bbox(paths[idx].split("!")[0], labels)



    
# Extract 
image_paths = labels["Frame"].unique()
np.random.shuffle(image_paths)
split_idx = int(image_paths.shape[0] * 0.80)

print("train index 0 ~ {} (size: {})".format(split_idx - 1, split_idx))
print("test index {} ~ {} (size: {})".format(split_idx,
                                             image_paths.shape[0] - 1, image_paths.shape[0] - split_idx))    


train_paths = image_paths[:split_idx]
test_paths = image_paths[split_idx:]
train_paths.shape, test_paths.shape


train_csv = labels[labels['Frame'].isin(train_paths)].reset_index(drop=True)
test_csv = labels[labels['Frame'].isin(test_paths)].reset_index(drop=True)


# only save these two columns
columns = ["Frame", "Mask"]

train_csv[columns].to_csv("pool_train.csv", index=False, header=False)
test_csv[columns].to_csv("pool_test.csv", index=False, header=False)

#train_csv[columns].to_csv("car_train.csv", index=False, header=False)
#test_csv[columns].to_csv("car_test.csv", index=False, header=False)









    