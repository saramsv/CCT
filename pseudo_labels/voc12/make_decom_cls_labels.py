#python3 make_decom_cls_labels.py
import json
import pandas as pd
import numpy as np
import argparse
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str)
    parser.add_argument("--out", default="decom_cls_labels_zyang.npy", type=str)
    args = parser.parse_args()


    df = pd.read_csv(args.train_list, sep = ":", names=('path', 'label'))


    classes = {'foot': 1, 'hand': 2, 'arm': 3, 'leg': 4, 'torso': 5, 'head':6} 
    #'foot': 1, 'hand': 2, 'arm': 3, 'leg': 4, 'torso': 5, 'body': 6, 'head':7}
    images_seen = set()
    d = dict()

    for index, row in df.iterrows():
        # get the image name and object
        path = row['path']
        if path not in images_seen: 
            images_seen.add(path)
            df_sub = df[df['path'] == path]

            label = np.zeros(len(classes))
            for index2 , row2 in df_sub.iterrows():
                try:
                    class_id = classes[row2['label']]
                    label[class_id - 1] = 1
                    print(row)
                except:
                    pass
                    #print("{} isn't in the label list".format(row2['label']))

            d[path] = label

    np.save(args.out, d)
