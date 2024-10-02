import os
import random
from pathlib import Path
import torchvision.transforms as T
from torchvision.io import read_image, write_jpeg

# Chemins des dossiers
input_folder = 'chemin/vers/ton/dossier/d_images'  # Dossier contenant les images originales
output_folder = 'chemin/vers/ton/dossier/de_sortie'  # Dossier pour enregistrer les images rognées

# Créer le dossier de sortie s'il n'existe pas
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Transformation de rognage aléatoire
crop_size = (224, 224)  # Taille de rognage
transform = T.RandomCrop(size=crop_size)

# Parcours des fichiers .jpg dans le dossier d'entrée
for idx, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith('.jpg'):
        # Chemin complet de l'image d'entrée
        img_path = Path(input_folder) / filename
        
        # Charger l'image
        img = read_image(str(img_path))
        
        # Appliquer la transformation de rognage
        cropped_img = transform(img)
        
        # Créer un nouveau nom de fichier avec un numéro
        new_filename = f"cropped_image_{idx + 1}.jpg"
        new_img_path = Path(output_folder) / new_filename
        
        # Enregistrer l'image rognée
        write_jpeg(cropped_img, str(new_img_path))
        
        print(f"Image {filename} rognée et enregistrée sous {new_filename}")
