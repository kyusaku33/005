import torch
from PIL import Image
import cv2
#from django.conf import settings

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
        
    if size is not None:

        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def load_image_style(filename, size=None, scale=None,max_style_size=1024):
    img = cv2.imread( filename)

    if img.ndim == 2:  # モノクロ
        pass
    elif img.shape[2] == 3:  # カラー
        img = img[:, :, ::-1]
    elif img.shape[2] == 4:  # 透過
        img = img[:, :, [2, 1, 0, 3]]
    img = Image.fromarray(img)

    print(img.size)
    # if img.size[0] < img.size[1]:
    #     img = img.rotate(-90)
        
    if int(img.size[0] ) > max_style_size:
        aspect = float(img.size[1])/float(img.size[0])
        img = img.resize((int(max_style_size), int(max_style_size*aspect)), Image.ANTIALIAS)

    if size is not None:

        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    print(img.size)
    return img



def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std
