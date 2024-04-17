import cv2
import imutils
from numpy import random, uint8

def add_gaussian_noise(img, mean = 0, std = 25):

    noise = random.normal(mean, std, img.shape).astype(uint8)
    noisy_img = cv2.add(img, noise)

    return noisy_img

def apply_noise_modif(img_file, folder_path):

    img = cv2.imread('flags/%s.png'%img_file)
    img = imutils.resize(img, width = 512)
    
    for i in range(1,11):
        std = round(10**(i/5))
        noisy_img = add_gaussian_noise(img,0,std)
    
        cv2.imwrite('{0}/{1}_noise_{2}.png'.format(folder_path,img_file,std), noisy_img)