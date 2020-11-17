import json
import pandas as pd
import sys
import os
import numpy as np



df = pd.read_csv(sys.argv[1], sep = ":", names=('path','label','is_annotated', 'pmi'))

data = {}

for index, row in df.iterrows():
    donor_id = row['path'].split("/")[-2]
    if donor_id not in data:
        all_imgs_df = df[df['path'].str.contains("sara_img/" + donor_id)]
        days = all_imgs_df['pmi'].values
        uniq_days = np.unique(days)
        for d in uniq_days:
             pmi_df = all_imgs_df[all_imgs_df['pmi'] == d]
             for i, row2 in pmi_df.iterrows():
                 label = row2['label']
                 if donor_id not in data:
                     data[donor_id] = {}
                 if str(d) not in data[donor_id]:
                     data[donor_id][str(d)] = {}
                 if label not in data[donor_id][str(d)]:
                     data[donor_id][str(d)][label] = {}
                 if row2['is_annotated'] not in data[donor_id][str(d)][label]:
                     data[donor_id][str(d)][label][row2['is_annotated']] = []
                 data[donor_id][str(d)][label][row2['is_annotated']].append(row2['path'])


for don in data.keys():
    for d in data[don].keys():
        all_ul = []
        all_l = []
        for bp in data[don][d]:
            if 'yes' in data[don][d][bp]:
                all_l.extend(data[don][d][bp]['yes'])
            if 'no' in data[don][d][bp]:
                all_ul.extend(data[don][d][bp]['no'])
        for i , img in enumerate(all_l):
            try:
                print(img, all_ul[i])
            except:
                print("ooops")


with open(sys.argv[1] + 'day_based.txt', 'w') as outfile:
    json.dump(data, outfile)

