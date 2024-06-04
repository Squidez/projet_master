import os
import re
import requests
from PIL import Image
from bs4 import BeautifulSoup

def main():
    
    #emplacement du fichier
    folder_path = input('Select folder path : ') 

    url = 'https://flagpedia.net/sovereign-states'  
    response = requests.get(url)

    # récupère toutes les balises img
    soup = BeautifulSoup(response.text, 'html.parser')
    images = soup.find_all('img')

    #Crée un dossier s'il n'existe pas
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    get_big_images(images, folder_path)
    print('flags loaded')

    for flag in os.listdir(folder_path):
        flag_path = f'{folder_path}/{flag}'
        img = Image.open(flag_path).convert('RGB')
        img.save(flag_path)

def get_small_images(images):
    """
    Fonctionne mais retourne des images de petites tailles puisqu'il s'agit de thumbnail
    """

    for image in images:

        link = 'https://flagpedia.net%s' % image['src']
        name = image['alt'].split()[2] # garde uniquemenet le nom du pays (Flag of ...)

        with open(name, 'wb+') as f:
            img = requests.get(link)
            f.write(img.content)

def get_big_images(images,folder_path):
    """
    Récupère les images des drapeaux en grand format (width = 2560px)
    """

    #Regex qui récupère l'abréviation des pays
    abreviation = r'(?<=\/)\w{2}(?=\.\w+)'
    #liste de toutes les abréviation
    abrev_list = [re.findall(abreviation, image['src'])[0] for image in images]
    
    for abrev in abrev_list:

        link = 'https://flagcdn.com/w2560/%s.png' % abrev
        
        with open('{0}/{1}.png'.format(folder_path, abrev), 'wb+') as f:
            img = requests.get(link)
            f.write(img.content)

if __name__ == "__main__":
    main()