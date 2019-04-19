# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:19:09 2019

@author: Amir.Khan
"""

import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
import numpy as np
from pathlib import Path

saver = tf.train.import_meta_graph("./models/model.ckpt.meta") #car trained model checkpoint
#pool trained model checkpoint
#saver = tf.train.import_meta_graph("./models_pool/model.ckpt.meta")

sess = tf.InteractiveSession()

saver.restore(sess, "models/model.ckpt") #car trained model checkpoint
#saver.restore(sess, "models_pool/model.ckpt")




X, mode = tf.get_collection("inputs")
pred = tf.get_collection("outputs")[0]



def plot_image(image, title=None, **kwargs):
    """Plots a single image

    Args:
        image (2-D or 3-D array): image as a numpy array (H, W) or (H, W, C)
        title (str, optional): title for a plot
        **kwargs: keyword arguemtns for `plt.imshow`
    """
    shape = image.shape

    if len(shape) == 3:
        pass
#        plt.imshow(image, **kwargs)
    elif len(shape) == 2:
        pass
#        plt.imshow(image, **kwargs)
    else:
        raise TypeError(
            "2-D array or 3-D array should be given but {} was given".format(shape))

    if title:
        plt.title(title)


def plot_two_images(image_A, title_A, image_B, title_B, figsize=(15, 15), kwargs_1={}, kwargs_2={}):
    """Plots two images side by side"""
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plot_image(image_A, title=title_A, **kwargs_1)

    plt.subplot(1, 2, 2)
    plot_image(image_B, title=title_B, **kwargs_2)
    

def plot_three_images(image_A, image_B, image_C, figsize=(15, 15)):
    """Plots three images side by side"""
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 3, 1)
    plot_image(image_A)
    
    plt.subplot(1, 3, 2)
    plot_image(image_B)
    
    plt.subplot(1, 3, 3)
    plot_image(image_C)
    
    
def read_image(image_path, gray=False):
    """Returns an image array

    Args:
        image_path (str): Path to image.jpg

    Returns:
        3-D array: RGB numpy image array
    """
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    image = cv2.imread(image_path)    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def already_drawn_bbox(bbox, left_top, right_bot):
    
    for (a, b), (c, d) in bbox:
        
        if a <= left_top[0] <= c:
            if a <= right_bot[0] <= c:
                if b <= left_top[1] <= d:
                    if b <= left_top[1] <= d:
                        return True
    
    return False

Total = []

def pipeline(image, threshold=0.9999, image_WH=(224, 224)):
    Total.clear()
    image = np.copy(image)
    H, W, C = image.shape
    
    if (W, H) != image_WH:
        image = cv2.resize(image, image_WH)
    
#    pred_scores = tf.nn.softmax(pred, axis=2)
#    print(pred_scores)
    
    mask_pred = sess.run(pred, feed_dict={X: np.expand_dims(image, 0),
                                          mode: False})    
    mask_pred = np.squeeze(mask_pred)
        
        
#    print(Total.append(mask_pred > threshold))
    mask_pred = mask_pred > threshold
  
    labeled_heatmap, n_labels = label(mask_pred)
       
    bbox = []
    
    for i in range(n_labels):
        mask_i = labeled_heatmap == (i + 1)
#        print(mask_i)
#        
#        for l in np.nditer(mask_i):
#            if l > threshold:
#                Total.append(l)
#        
#        cf_score.append(Average(Total))
#    
#        Total.clear()
        
        nonzero = np.nonzero(mask_i)
        
        nonzero_row = nonzero[0]
        nonzero_col = nonzero[1]
        
        left_top = min(nonzero_col), min(nonzero_row)
        right_bot = max(nonzero_col), max(nonzero_row)
      
        
        if not already_drawn_bbox(bbox, left_top, right_bot):
            image = cv2.rectangle(image, left_top, right_bot, color=(0, 255, 0), thickness=3)
            bbox.append((left_top, right_bot))

        
    Total.append(bbox)
   
    return image


test_data = pd.read_csv("test_labels.csv")

idx_list = [3]

for idx in idx_list:
    image_path = test_data.iloc[idx]["Frame"]
    base = Path(image_path).resolve().stem   
    image = read_image(image_path)
    plot_two_images(image, "original", pipeline(image), "prediction")
    with open("pool_pred"+"/"+base+".txt", "w") as f:
        for b in Total:
            for l in b:
                dataline = str(l[0][1]) + " " + str(l[0][0]) + " " + str(l[1][1]) + " " + str(l[1][0])
                f.write("2" +" "+ "0.99" + " " +dataline + '\n')



#Make Predictions File
for idx in idx_list:
    image_path = test_data.iloc[idx]["Frame"]
    base = Path(image_path).resolve().stem  
    with open("pool_pred"+"/"+base+".txt") as f:
        lines = f.readlines()
        lines = [l for l in lines]
        with open("car_pred"+"/"+base+".txt", "a") as f1:
            items = map(lambda x: x, lines)
            f1.writelines(items)















        