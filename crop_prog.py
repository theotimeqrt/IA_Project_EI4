import os
from random import randint
from pathlib import Path
import torchvision.transforms as T
from torchvision.io import read_image, write_jpeg
import shutil
from PIL import Image

new_filename_avant = "image99_cropped"

# Chemins des dossiers
folder_path = './images/images_originales'  # Dossier contenant les images originales
output_folder = './images/images_sorties'  # Dossier pour enregistrer les images rognées
text_path = './images/text_infos'   # Dossier pour enregistrer le ficher '.txt'

# Vider le dossier contient les images coupées
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
    
if os.path.exists(text_path):
    shutil.rmtree(text_path)

# Créer les dossiers nécessaires s'ils n'existent pas
Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(text_path).mkdir(parents=True, exist_ok=True)


# Parcours des fichiers .jpg dans le dossier d'entrée
for idx, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith('.jpg'):
        # Chemin complet de l'image d'entrée
        img_path = Path(folder_path) / filename

        # 获取原始文件名（不带扩展名）和扩展名
        original_name, ext = os.path.splitext(filename)
        
        # Charger l'image
        img = read_image(str(img_path))

        img_PIL = Image.open(img_path)
        image_tensor = T.functional.to_tensor(img_PIL)

        # Les dimensions du tenseur seront de la forme (canaux, hauteur, largeur)
        _, height, width = image_tensor.shape
        print("height=",height,"width=",width)

        # Transformation de rognage aléatoire
        random_size_w = int(0.1 * width + randint(0, int(0.8 * width)))
        random_size_h = int(0.1 * height + randint(0, int(0.8 * height)))
        crop_size = (random_size_h, random_size_w)  # Taille de rognage
        transform = T.RandomCrop(size=crop_size)

        # Appliquer la transformation de rognage
        cropped_img = transform(img)
        
        # Créer un nouveau nom de fichier avec un numéro
        new_filename = f"{original_name}_cropped"
        new_img_path = Path(output_folder) / new_filename
        
        # Enregistrer l'image rognée
        write_jpeg(cropped_img, str(new_img_path))

        with open('./images/text_infos/images_coupees.txt', 'a') as f:
            f.write(f"{original_name} {new_filename} 1\n")

        with open('./images/text_infos/images_coupees.txt', 'a') as f:
            f.write(f"{original_name} {new_filename_avant} 0\n")

        new_filename_avant = new_filename 

# Copier tous les images originales au dossier_couped
files = os.listdir(folder_path)
    
# Copier seulement les fichers '.jpg'
for file in files:
    if file.lower().endswith('.jpg'):
        src_file = os.path.join(folder_path, file)
        dest_file = os.path.join(output_folder, file)

        # Copier les images 
        shutil.copy2(src_file, dest_file)
        
        print(f"Image {filename} rognée et enregistrée sous {new_filename}")

