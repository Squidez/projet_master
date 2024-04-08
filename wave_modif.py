from PIL import Image
from matplotlib import pyplot
# import torch
from torchvision.transforms import v2

# Image test (Afrique du sud)
img = Image.open('flags/za.png').convert('RGBA')
img.thumbnail((1080,180))
alpha = range(25,251, 25)
pyplot.ion()

for a in alpha:

    elastic_transformer = v2.ElasticTransform(alpha=a)
    img2 = elastic_transformer(img)

    pyplot.imshow(img2)
    pyplot.pause(0.005)
    
    next = input('alpha {0}'.format(a))


