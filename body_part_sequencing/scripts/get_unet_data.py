import os


data = open('/data/sara/CCT/body_part_data/data/unet_data', 'r').readlines()

for l in data:
    img_name = l.split()[0].strip().split('/')[-1].replace('.JPG', '')
    if os.path.isfile("/data/sara/CCT/body_part_data/labels/"+ img_name + '.png'):
        print(l.split()[0] + " /data/sara/CCT/body_part_data/labels/" + img_name + ".png")
    elif os.path.isfile("/data/sara/CCT/body_part_data/val_imgs/"+ img_name + '.png'):
        print(l.split()[0] + " /data/sara/CCT/body_part_data/val_imgs/" + img_name + '.png')
