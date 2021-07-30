# Create a json to directly read in Yolo_dataset in dataset_UAV.py
import os
import random
import sys
import pickle

_DATA_DIR = '/home/shrey/vecros/data/UAV-benchmark-M'
SAVE_FILENAME='traindata'

train_images = os.path.join(_DATA_DIR,'train','seq')
train_gt = os.path.join(_DATA_DIR,'train','gt')
truth = {}
for seq in os.listdir(train_images):
    gt_filename=seq+'_gt_whole.txt'
    lable_path=os.path.join(train_gt,gt_filename)
    print(lable_path)
    
    for img in os.listdir(os.path.join(train_images,seq)):
        img_name = seq+'_'+img
        img_numb = abs(int("".join(filter(str.isdigit, img.split(".")[0]))))
        f = open(lable_path, 'r', encoding='utf-8')
        truth[img_name] = []
        for line in f.readlines():
            data = line.split(",")
              #data[0] = img_name(M010_IMG00003.JPG) search for 3rd frame in gt file in a loop
            #print(data[0])
            
            if int(data[0])==img_numb:
                
                bbox = [int(float(data[2])),int(float(data[3])),int(float(data[4])),int(float(data[5]))]
                truth[img_name].append(bbox)

print(len(truth))
with open(SAVE_FILENAME + '.pickle', 'wb') as f:
    pickle.dump(truth, f, pickle.HIGHEST_PROTOCOL)

    