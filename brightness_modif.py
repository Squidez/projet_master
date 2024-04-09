from PIL import Image, ImageEnhance
from matplotlib import pyplot

# Image test (Afrique du sud)
img = Image.open('flags/za.png').convert('RGBA')
img.thumbnail((1080,1080))

pyplot.ion()

enhancer = ImageEnhance.Brightness(img)

for i in range(1,11):
    
    # Si le facteur i est en dessous de 1 l'image est sous exposé, et inversemment
    factor = i/4.5
    img2 = enhancer.enhance(factor)

    pyplot.imshow(img2)
    pyplot.pause(0.005)
    
    next = input('factor {0}'.format(factor))