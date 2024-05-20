from PIL import Image, ImageEnhance

def apply_brightness_modif(img_file, folder_path):

    img = Image.open('flags/%s.png'%img_file).convert('RGBA')
    img.thumbnail((512,512))

    enhancer = ImageEnhance.Brightness(img)
    factor = [0.05, 0.2,0.5,1.5,2,3,4,6,8,10]
    for f in factor:
        # Si le facteur i est en dessous de 1 l'image est sous exposé, et inversemment
        bright_img = enhancer.enhance(f)
        bright_img.save('{0}/{1}_bright_{2}.png'.format(folder_path,img_file,f))