from PIL import Image
# import torch
from torchvision.transforms import v2
    
def apply_wave_modif(img_file, folder_path):

    # Image test (Afrique du sud)
    img = Image.open('flags/%s.png'%img_file).convert('RGBA')
    img.thumbnail((512,512))

    for a in range(1,11):
        # modification du param exponentiel pour avoir des résultat plus "extrème"
        alpha = a**3

        # Applique la déformation suivant le niveau alpha
        elastic_transformer = v2.ElasticTransform(alpha=alpha)
        wavey_img = elastic_transformer(img)
        wavey_img.save('{0}/{1}_wave_{2}.png'.format(folder_path,img_file,alpha))
    

