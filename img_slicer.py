import os
import cv2
import argparse
import random 
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Code for splitting into smaller images')
parser.add_argument('--input', help='Path to the first input image.', default='result.png')
parser.add_argument('--labels', help='Masked labels image', default='inklabels.png')
parser.add_argument('--mask', help='Mask for borders', default='mask.png')

parser.add_argument('--trainpath', help='Path to the first input image.', default='./data/train/')
parser.add_argument('--validpath', help='Masked image', default='./data/valid/')

parser.add_argument('--inpath', help='Path to the first input image.', default='./images/')
parser.add_argument('--maskpath', help='Masked image', default='./masks/')

parser.add_argument('--train_sample', help='Fraction of images for training', default='0.8')

parser.add_argument('--w', help='Width of cropped split image', default='256')
parser.add_argument('--h', help='Width of cropped split image', default='256')

#for testing if mask is aligned
parser.add_argument('--masktestpath', help='Path for testing alignment of mask', default='./masks_test/')

args = parser.parse_args()

#Cropped training data size
crop_size = np.array([int(args.w),int(args.h)])

#Image
img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

#Mask
mask = cv2.imread(args.labels, cv2.IMREAD_GRAYSCALE)

#Mask border, dont generate here
mask_border = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

#IR test mask
test_mask = cv2.imread("ir.png", cv2.IMREAD_GRAYSCALE)


if img is None:
    print('Could not open or find the image: ', img)
    exit(0)

if mask is None:
    print('Could not open or find the image: ', mask)
    exit(0)


print(img.shape[:2])
height, width = img.shape[:2]

print("Input Image height: %dpx, width: %dpx" %(height,width))

margin = (crop_size/2).astype(int)

width_lim  = np.array([margin[0],width-margin[0]-1])  #width  limits
height_lim = np.array([margin[1],height-margin[1]-1]) #height limits

N_sample = 1000

print(width_lim)
print(height_lim)

#random center pixel
random_px = []
for i in range(0,N_sample):
    random_px.append([random.randint(width_lim[0],width_lim[1]),
                      random.randint(height_lim[0],height_lim[1])])

random_px = np.array(random_px)

#Create folder
def create_dir(path):
     if not os.path.exists(path):
         os.makedirs(path)

#create sample directories
#train folder
create_dir(args.trainpath + args.inpath)
create_dir(args.trainpath + args.maskpath)
create_dir(args.trainpath + args.masktestpath)
#validation folder
create_dir(args.validpath + args.inpath)
create_dir(args.validpath + args.maskpath)
create_dir(args.validpath + args.masktestpath)

#Split train and validation sample by fraction defualt 80% train 20% valid
N_train = int(float(args.train_sample)*N_sample)
N_valid = N_sample - N_train

#Generate training sample
num = 0
while num < N_train:
    center_px = [random.randint(width_lim[0],width_lim[1]),
                 random.randint(height_lim[0],height_lim[1])]
    
    w_crop = np.array([center_px[0]-margin[0],center_px[0]+margin[0]]) #width crop
    h_crop = np.array([center_px[1]-margin[1],center_px[1]+margin[1]]) #height crop
    
    crop_input = img[h_crop[0]:h_crop[1],w_crop[0]:w_crop[1]]
    crop_mask  = mask[h_crop[0]:h_crop[1],w_crop[0]:w_crop[1]]

    crop_mask_border = mask_border[h_crop[0]:h_crop[1],w_crop[0]:w_crop[1]]
    crop_test_mask   = test_mask[h_crop[0]:h_crop[1],w_crop[0]:w_crop[1]]
    fraction = crop_mask_border[crop_mask_border > 0].size/crop_mask_border.size

    #Makes sure that we are at least 50% within the borders of the fragment
    #Don't need to train on the known dark areas with no inputs
    if fraction > .5:
        num += 1
        cv2.imwrite(args.trainpath + args.inpath   + "%d.png" % num, crop_input)
        cv2.imwrite(args.trainpath + args.maskpath + "%d.png"  % num, crop_mask)
        cv2.imwrite(args.trainpath + args.masktestpath + "%d.png"  % num, crop_test_mask)
    else:
        continue
    
#Generate valid sample
num = 0
while num < N_valid:
    center_px = [random.randint(width_lim[0],width_lim[1]),
                 random.randint(height_lim[0],height_lim[1])]
    
    w_crop = np.array([center_px[0]-margin[0],center_px[0]+margin[0]]) #width crop
    h_crop = np.array([center_px[1]-margin[1],center_px[1]+margin[1]]) #height crop
    
    crop_input = img[h_crop[0]:h_crop[1],w_crop[0]:w_crop[1]]
    crop_mask  = mask[h_crop[0]:h_crop[1],w_crop[0]:w_crop[1]]

    crop_mask_border = mask_border[h_crop[0]:h_crop[1],w_crop[0]:w_crop[1]]
    crop_test_mask   = test_mask[h_crop[0]:h_crop[1],w_crop[0]:w_crop[1]]
    fraction = crop_mask_border[crop_mask_border > 0].size/crop_mask_border.size

    #Makes sure that we are at least 50% within the borders of the fragment
    #Don't need to train on the known dark areas with no inputs
    if fraction > .5:
        num += 1
        cv2.imwrite(args.validpath + args.inpath   + "%d.png" % num, crop_input)
        cv2.imwrite(args.validpath + args.maskpath + "%d.png"  % num, crop_mask)
        cv2.imwrite(args.validpath + args.masktestpath + "%d.png"  % num, crop_test_mask)
    else:
        continue

