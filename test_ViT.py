import torch
import torchvision.transforms as T
from torchvision.models import vit_b_16
# ici modèle base (moins fort que large et 16x16 bits de patches)
from PIL import Image

# Charger le modèle ViT pré-entraîné
model = vit_b_16(pretrained=True)
model.eval()

# Charger et transformer les images 
transform = T.Compose([
    T.Resize((224, 224)), # nouvelles dimensions, propre au modèle
    T.ToTensor(), # converti en tenseur, pixels attribué entre 0 et 1 les couleurs
    # Normalisation supplémentaire nombres sont standard de référence pour entrainement modèles moyenne = [R,G,B] puis ecart type.
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Charger les images gràce aux chemins
image1 = Image.open("images/I1.jpg")
image2 = Image.open("images/test_image.jpg")

# Partie crop 
#transform = v2.RandomCrop(size=(224, 224))
#out = transform(img)

#plot([img, out])

# utilisation transform (redimensionner) . dimension batch 
# un "batch" de l'image est souvent attendu par les modèles, on rajoute au tenseur l'indication qu'il n'y a qu'une image 
input1 = transform(image1).unsqueeze(0)
input2 = transform(image2).unsqueeze(0)

# Faire passer les entrées dans les modèles 
with torch.no_grad(): # désactive calcul des gradients, pas apprentissage en cours
    output1 = model(input1)
    output2 = model(input2)

# Afficher les sorties pour voir pourquoi on a 1 parfois 
# print("Output1:", output1)
# print("Output2:", output2)

# Calculer la similarité entre les deux images 
similarity = torch.nn.functional.cosine_similarity(output1, output2)
print(f"Similarity score : {similarity.item()}")

# Petite analyse
if similarity > 0.9:
    print("very good")
elif similarity > 0.7:
    print("good")