#python3 fix_tags.py polygon_annotated_tags
import json
import pandas as pd
import sys
import os



df = pd.read_csv(sys.argv[1], delimiter = ',')


#classes = {'foot' = 10, 'hand' = 20, 'arm' = 30, 'leg' = 40, 'torso'= 50, 'body' = 60}
classes = ['head', 'foot', 'hand', 'arm', 'leg', 'torso']


def clean_tag(row):
    tag = row['tag']
    for c in classes:
        if c in tag:
            row['tag'] = c
            return row
print("_id,user,location,image,tag,created,__v")
i = 0
for index, row in df.iterrows():
    row = clean_tag(row)
    line = ''
    for i in row.values:
        line += str(i)
        line += ','
    print(line)

