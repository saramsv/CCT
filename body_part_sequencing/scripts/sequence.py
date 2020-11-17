import sys
from scipy import spatial
import numpy as np
import csv
import ast
import datetime
import math


def key_func(x):
    date_ = x.split('/')[-1]
    y = '00'
    if date_[3] == '1':
        y = '12'
    elif date_[3] == '0':
        y = '11'
    m = date_[4:6]
    d = date_[6:8]
    if d == '29' and m == '02':
        d = '28'
    date_ = m + d + y
    return datetime.datetime.strptime(date_, '%m%d%y')
#########################################################
def cosine_similarity(v1,v2):
    return 1 - spatial.distance.cosine(v1, v2)
##########################################################
def overlap_merge(all_sims):
    no_more_merge = False
    while no_more_merge == False:
        merged_dict = {}
        seen = []
        all_sims_keys = list(all_sims.keys())
        no_more_merge = True
        for key1 in all_sims_keys:
            if key1 not in seen:
                if key1 not in merged_dict :
                    merged_dict[key1] = list(set(all_sims[key1]))#to remove the duplicates
                for key2 in all_sims_keys:
                    if key1 != key2:
                        intersect = len(set(all_sims[key1]).intersection(set(all_sims[key2])))
                        if intersect != 0:
                            no_more_merge = False
                            merged_dict[key1].extend(list(set(all_sims[key2])))
                            merged_dict[key1] = sorted(merged_dict[key1], key = key_func)
                            seen.append(key2)
        all_sims = merged_dict
    return all_sims 
##########################################################################
def add_to_similarity_dict(all_sims, similarities, key, count, mean_sim):#, ratio):
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    max_ = similarities[0][1]
    mean_sim = mean_sim * (count - 1) + max_
    mean_sim = mean_sim / count
    threshold = max(0.99 * max_, mean_sim)
    '''
    print(similarities)
    print("thre: {}".format(threshold))
    print("key: {}".format(key))
    if 0.95 * max_ > mean_sim:
        threshold = mean_sim
    else:
        threshold = 0.95 * max_ # instead of mean_sim I had a constant like 0.92
    print("max: {}, mean: {}, threshold: {} ".format(max_, mean_sim, threshold))
    '''
    if key not in all_sims:
        all_sims[key] = [key]
    for pair in similarities:
        if pair[1] >= threshold:
            all_sims[key].append(pair[0])
    return all_sims, mean_sim


##################################################################
def print_(all_sims, donor):
    label = 0
    not_sequenced = []
    print(len(all_sims))
    with open("../data/sequences/" + donor + "_pcaed_sequenced", 'w') as f_seq:
        for key in all_sims:
            if len(all_sims[key]) > 1:
                label = label + 1
                for img in all_sims[key]:
                    temp = img.replace('JPG', 'icon.JPG: ')
                    #print(temp + donor + "_" + str(label))
                    f_seq.write(temp + donor + "_" + str(label) + "\n")
            else:
                not_sequenced.append(all_sims[key])
    with open("../data/sequences/" + donor + "_not_sequenced", 'w') as f:
        for image in not_sequenced: 
            f.write(image[0] + "\n")
#################################################################
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
################################################################
def match(day1, day2, all_sims, count, mean_sim, donor2day2img, all_embs, donor):
    day1_imgs = donor2day2img[donor][day1]
    for day1_img in day1_imgs:
        emb = all_embs[day1_img]
        key = day1_img
        for seen in all_sims:
            for x in all_sims[seen]:
                if day1_img ==  x: # if it is one of the matched ones
                    key = seen
            
        day2_imgs = donor2day2img[donor][day2]
        similarities = []
        for day2_img in day2_imgs:
            emb2 = all_embs[day2_img] 

            sim = cosine_similarity(emb, emb2)
            similarities.append([day2_img, sim])
        count += 1
        #print(day1_img)
        all_sims, mean_sim = add_to_similarity_dict(all_sims, similarities, key, count, mean_sim)
        '''
        if day1_img == '/home/mousavi/da1/icputrd/arf/mean.js/public/sara_img/07b/07b00128.04.JPG':
            import bpython
            bpython.embed(locals())
        '''
    return all_sims, mean_sim, count
#################################################################
def sequence_finder(donor2img2embeding, donor2day2img):
    for donor in donor2img2embeding:
        days = list(donor2day2img[donor].keys())
        days.sort()
        all_embs = donor2img2embeding[donor]
        all_sims = {} #key = imgs, value = [[im1, dist],im2, dit[],...]
        window_size = 4
        compared = []
        mean_sim = 0
        count = 0
        windows = rolling_window(np.array(range(len(days))), window_size)
        for window in windows:
            for ind1 in range(len(window)):
                for ind2 in range(ind1 + 1, len(window)):
                    pair = (window[ind1], window[ind2])
                    if pair not in compared:
                        compared.append(pair)
                        day1_ind = pair[0]
                        day2_ind = pair[1]
                        day1 = days[day1_ind]
                        day2 = days[day2_ind]
                        #import bpython
                        #bpython.embed(locals())
                        all_sims, mean_sim, count = match(day1, day2, all_sims, count, mean_sim, donor2day2img, all_embs, donor)
                        #print(all_sims)
                        #_ = input()

        all_sims, mean_sim, count = match(day2, day1, all_sims, count, 
                mean_sim, donor2day2img, all_embs, donor)
       # all_sims = overlap_merge(all_sims)

        print_(all_sims, donor)
