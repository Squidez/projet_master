from PIL import Image, ImageFilter
from matplotlib import pyplot
import cv2
from numpy import random, uint8

## TEST AVEC PIL
# # Image test (Afrique du sud)
# img = Image.open('flags/za.png').convert('RGBA')

# pyplot.ion()

# for i in range(1,11):

#     img2 = img.filter(ImageFilter.GaussianBlur(i*4))

#     pyplot.imshow(img2)
#     pyplot.pause(0.005)
    
#     next = input('next')

#TEST AVEC OPENCV

def add_gaussian_noise(img, mean = 0, std = 25):
    noise = random.normal(mean, std, img.shape).astype(uint8)
    noisy_img = cv2.add(img, noise)

    return noisy_img

img = cv2.imread('flags/za.png')
img = cv2.resize(img, None, fx = 0.3,fy=0.3)
pyplot.ion()

for i in range(1,11):

    std = 10**(i/4)
    noisy_img = add_gaussian_noise(img,0,std)

    pyplot.imshow(noisy_img)
    pyplot.pause(0.005)

    next = input('std = {0}'.format(std))