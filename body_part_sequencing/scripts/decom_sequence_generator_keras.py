from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import sys
import numpy as np
import datetime
import argparse
from keras.models import load_model, Sequential
from keras.applications.vgg16 import preprocess_input
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
import keras
import sequence
import cv2
import json



def key_func(x):
    try:
        date_ = x.split('/')[-1]
        y = '00'
        if date_[3] == '1':
            y = '12'
        elif date_[3] == '0':
            y = '11'
        m = date_[4:6]
        d = date_[6:8]
        date_ = m + d + y
        return datetime.datetime.strptime(date_, '%m%d%y')
    except:
        import bpython
        bpython.embed(locals())

        

def sort_dates(donors2imgs): #sorts the dates by getting a list of img_names for each donor and sorting that
    for key in donors2imgs:
        donors2imgs[key] = sorted(donors2imgs[key], key=key_func)
    return donors2imgs


def cal_day_from_deth(donors2imgs_sorted):
    for key in donors2imgs_sorted:
        day2imgs = {} 
        first_img = True
        for img in donors2imgs_sorted[key]:
            if first_img == True:
                start_time = key_func(img)
                first_img = False
            img_time = key_func(img)
            time_from_start = (img_time - start_time).days
            if time_from_start not in day2imgs:
                day2imgs[time_from_start] = []
            day2imgs[time_from_start].append(img)
        donors2imgs_sorted[key] = day2imgs 
    return donors2imgs_sorted 
    # this a dictionary with each donor_id as keys and values are another 
    #dictionary with keys being xth days since day one and the values are a 
    #list of images that belong to day xth for that donor.

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type = str)
    parser.add_argument('--config', default='config.json', type = str)

    args = parser.parse_args()

    paths = args.paths
    config = json.load(open(args.config))
    model_path = config['resnet_weigth_path']
    
    model_no_top = Sequential()

    model_no_top = ResNet50(include_top = False, pooling='ave', weights = 'imagenet')
    #model_no_top.layers[0].trainable = False
    model_no_top.trainable = False


    model_no_top.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    #model = load_model(model_path)
    #model_no_top = keras.models.Sequential(model.layers[:-3])


    donors2imgs = {}
    donors2img2embed = {}
    imgname2add = {}

    image_paths = open(paths).readlines()
    img_size = 224 
    not_found = 0
    for row in image_paths:
        try:
            img_name = row.strip()
            img_object = cv2.imread(img_name)
            img_object = cv2.resize(img_object, (img_size, img_size))
            img_object = np.array(img_object, dtype = np.float64)
            img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))

            feature = model_no_top.predict(img_object)[0][0][0]
            #feature = np.array(feature)
            donor_id = img_name.split("/")[-2]
            if donor_id not in donors2img2embed and donor_id not in donors2imgs:
                donors2img2embed[donor_id] = {} 
                donors2imgs[donor_id] = [] # a list for all of the images belonging to the same donor
            donors2img2embed[donor_id][img_name] = feature 
            # this a dictionary with each donor_id as keys and values are another dictionary
            # with keys being an image and the values being the feature vector for that imag

            donors2imgs[donor_id].append(img_name)
        except:
            not_found += 1
    donors2imgs_sorted = sort_dates(donors2imgs) # this sorts the images for a donor based on their dates
    donor2day2imgs = cal_day_from_deth(donors2imgs_sorted)

    day2clus2emb = sequence.sequence_finder(donors2img2embed, donor2day2imgs) 

