from PIL import Image
# import torch
from torchvision.transforms import v2
    
def apply_wave_modif(img_file, folder_path):

    
    img = Image.open('flags/%s.png'%img_file).convert('RGB')
    img.thumbnail((512,512))

    for a in range(1,11):
        # modification du param exponentiel pour avoir des résultat plus "extrème"
        alpha = a**3

        # Applique la déformation suivant le niveau alpha
        elastic_transformer = v2.ElasticTransform(alpha=alpha)
        wavey_img = elastic_transformer(img)
        wavey_img.save('{0}/{1}_wave_{2}.png'.format(folder_path,img_file,alpha))

img = Image.open('flags/za.png').convert('RGB')
img.thumbnail((256 ,256))

bright_transformer = v2.ColorJitter(brightness=1, contrast=1, saturation=0.4)
rotation_transformer = v2.RandomRotation(degrees=(-20,20))
blur_transformer = v2.GaussianBlur(kernel_size=1, sigma=(0.1, 5))
perspective_transform = v2.RandomPerspective(distortion_scale=0.5, p=1)
elastic_transformer = v2.ElasticTransform(alpha=125)

for i in range(1,11):
    bright_img = blur_transformer(img)
    bright_img.show()
    

