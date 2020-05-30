import os
from PIL import Image
import numpy as np

def getImagesWithID(path):
    h = 150
    w = 150
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        faceToResize = Image.open(imagePath).convert('L')
        faceImg = faceToResize.resize((h,w), Image.ANTIALIAS)
        faceNp = np.array(faceImg, 'uint8')
        faceNp = faceNp.flatten()
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        Ids.append(ID)
    return np.array(Ids), np.array(faces), h, w
