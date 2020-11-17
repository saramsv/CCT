#! /user/bin python3
import sys

file_ = sys.argv[1]

data = open(file_, 'r').readlines()
for line in data:
    if line.split(":")[1].strip() =='fullbody':
        print(line.split(":")[0].strip() + ":" + "legs")
        print(line.split(":")[0].strip() + ":" + "foot")
        print(line.split(":")[0].strip() + ":" + "arm")
        print(line.split(":")[0].strip() + ":" + "hand")
        print(line.split(":")[0].strip() + ":" + "torso")
        print(line.split(":")[0].strip() + ":" + "head")
    else:
        print(line.strip())
