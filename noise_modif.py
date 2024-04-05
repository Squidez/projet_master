from PIL import Image, ImageFilter
from matplotlib import pyplot

# Image test (Afrique du sud)
img = Image.open('flags/za.png').convert('RGBA')

pyplot.ion()

for i in range(1,11):

    img2 = img.filter(ImageFilter.GaussianBlur(i*4))

    pyplot.imshow(img2)
    pyplot.pause(0.005)
    
    next = input('next')