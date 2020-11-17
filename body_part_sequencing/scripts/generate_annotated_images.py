from shapely.geometry import Point
from shapely.geometry import Polygon
import json
import pandas as pd
import sys
import cv2
import os
import numpy as np



df = pd.read_csv(sys.argv[1],  delimiter = ',', names = ['_id', 'user', 'location', 'image', 'tag', 'created', '__v'])
new_size = 400 # it means the images will be 224*224

def read_img(path):
    flag = True
    path = "/home/mousavi/da1/icputrd/arf/mean.js/public/" + path
    #path = "/data/sara/image-segmentation-keras/test_imgs/" + path
    if os.path.isfile(path) == False:
        print("this image does not exist:" , path)
        flag = False
    img_obj = cv2.imread(path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    return img_obj, flag


classes = {'foot': 1, 'hand': 2, 'arm': 3, 'leg': 4, 'torso': 5, 'head':6}
#colors = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128)]
colors = [(0, 0, 128), (0, 128, 0), (0, 128, 128), (128, 0, 0), (128, 0, 128), (128, 128, 0)]

print(classes)
images_seen = set()

for index, row in df.iterrows():
    # get the image name and object
    path = row['image']
    name = path.split('/')[-1]
    if name not in images_seen: 
        images_seen.add(name)
        img, flag = read_img(path)

        #get image size and do the resizing
        if flag:
            height, width = img.shape[:2]
            print(width, height)
            if  width <= height:
                width_percent = new_size / float(width)
                new_height = int(float(height) * float(width_percent))
                new_width = new_size
                img = cv2.resize(img, (new_width, new_height))
            elif height <= width:
                height_percent = new_size /float(height)
                new_width = int(float(width) * float(height_percent))
                new_height = new_size
                img = cv2.resize(img, (new_width, new_height)) 
            #orig_img_name = path.split("/")[-1]
            #cv2.imwrite("/data/sara/Imgs/" + orig_img_name , img)
            ann_img = np.zeros((img.shape[0],img.shape[1], 3)).astype('uint8') 
            #create an empty image
            #ann_img = np.ones((img.shape[1],img.shape[0], 3)).astype('uint8') #create an empty image
            #ann_img *= 255
            height, width = ann_img.shape[:2]
            #Find all of the tags for this image
            print(width, height)
            df_sub = df[df['image'] == path]

            polygons = []

            for index2 , row2 in df_sub.iterrows():
                location = row2['location']
                loc = json.loads(location)
                geometry = loc[0]['geometry'] #get the whole geomety section od the coordinate
                geometry_points = geometry['points']# get the points of the geometry. This is a list of points(x, y) = [{}, {}, ...]
                polygon_points = [] #this will hold the points that shape the polygon for us
                class_id = classes[row2['tag']]
                #print(class_id)
                for p in geometry_points: # access each point to convert it from ratio to numbers 
                    x = p['x'] # x is ratio and needs to be converted to actual number
                    x = x * width
                    y = p['y']#y is ratio and needs to be converted to actual number
                    y = y * height
                    polygon_points.append((x, y))
                polygon = Polygon(polygon_points)
                polygons.append((class_id, polygon))


            #for each pixel:
            for h in range(height):
                for w in range(width):
                    for class_id, polygon in polygons:
                        p = Point(w, h)
                        if polygon.contains(p):
                            ann_img[h,w] = colors[class_id-1] # class_id #
                            break


            #name2 = "/data/sara/image-segmentation-keras/test_annotations/"+ name.replace('.JPG', ".png")
            name2 = name.replace('.JPG', ".png")
            cv2.imwrite(name2, ann_img)
