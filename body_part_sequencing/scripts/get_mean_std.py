import sys
import math
import csv
import numpy as np
from tensorflow.keras.preprocessing import image

R_channel = 0
G_channel = 0
B_channel = 0

R_total = 0 
G_total = 0
B_total = 0

base_model_img_size = 224
total_pixel = 0

images = []
with open('body_part_imgs', 'r') as file_:
    csv_reader = csv.reader(file_)
    num_imgs = 0
    for row in csv_reader:
        img = image.load_img(row[0].strip(),
                target_size = (base_model_img_size,
                 base_model_img_size, 3), grayscale = False)
        images.append(row[0].strip())
        img = image.img_to_array(img)
        R_channel += np.sum(img[:,:,0])
        G_channel += np.sum(img[:,:,1])
        B_channel += np.sum(img[:,:,2])
        num_imgs += 1
    num = num_imgs * base_model_img_size * base_model_img_size
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num




    for im in images:
        img = image.load_img(im,
                target_size = (base_model_img_size,
                 base_model_img_size, 3), grayscale = False)
        img = image.img_to_array(img)
        total_pixel += img.shape[0] * img.shape[1]

        R_total += np.sum((img[:, :, 0] - R_mean) ** 2)
        G_total += np.sum((img[:, :, 1] - G_mean) ** 2)
        B_total += np.sum((img[:, :, 2] - B_mean) ** 2)

    R_std = math.sqrt(R_total / total_pixel)
    G_std = math.sqrt(G_total / total_pixel)
    B_std = math.sqrt(B_total / total_pixel)
    print("mean: {}, {}, {}".format(R_mean, G_mean, B_mean))
    print("std: {}, {}, {}".format(R_std, G_std, B_std))
    import bpython
    bpython.embed(locals())

