import argparse
import os
import sys
import time
import re

#import sqlite3
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from django.conf import settings

#import .utils
from .utils import load_image,load_image_style,save_image,gram_matrix,normalize_batch

from .transformer_net import TransformerNet
from .vgg import Vgg16
from img_trans.models import ImageUpload

def check_paths(save_model_dir,checkpoint_model_dir):
    try:
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        if checkpoint_model_dir is not None and not (os.path.exists(checkpoint_model_dir)):
            os.makedirs(checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(cuda,seed,image_size,dataset,batch_size,lr,style_image,style_size,epochs,content_weight,style_weight,log_interval,checkpoint_model_dir,checkpoint_interval,save_model_dir):
    device = torch.device("cuda" if cuda else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = load_image(style_image, size=style_size)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(y) for y in features_style]

    for e in range(epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if checkpoint_model_dir is not None and (batch_id + 1) % checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        content_weight) + "_" + str(style_weight) + ".model"
    save_model_path = os.path.join(save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(content_image,content_scale,output_image,model,cuda,export_onnx,max_style_size):
    device = torch.device("cuda" if cuda else "cpu")

    content_image = load_image_style(content_image, scale=content_scale, max_style_size=max_style_size)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():

            style_model = TransformerNet()
            state_dict = torch.load(model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if export_onnx:
                assert export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, export_onnx).cpu()
            else:
                
                output = style_model(content_image).cpu()
    
    save_image(output_image, output[0])


def stylize_onnx_caffe2(content_image, args):
    """
    Read ONNX model and run it using Caffe2
    """

    assert not export_onnx

    import onnx
    import onnx_caffe2.backend

    model = onnx.load(model)

    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)


def main(subcommand,pk,style,param,input_path):

    # con = sqlite3.connect(str(settings.BASE_DIR)+ '/db.sqlite3')  
    # c = con.cursor()
    # c.execute('SELECT * from img_trans_imageupload  WHERE id = "{}";'.format(pk))
    # db = c.fetchone() 

    db = ImageUpload.objects.get(pk=pk)
    #path = db.filse
    #con.commit()
    #con.close() 

    #subcommand = "eval"
    #style = "candy"
    epoch = 2
    batch_size = 4
    dataset = str(settings.BASE_DIR) + str(settings.STATIC_URL) + "style_image/train-images"
    style_image = str(settings.BASE_DIR) + str(settings.STATIC_URL) + "style_image/style-images"
    save_model_dir = str(settings.BASE_DIR) + str(settings.STATIC_URL) + "style_image/saved_models"
    checkpoint_model_dir = str(settings.BASE_DIR) + str(settings.STATIC_URL) + "style_image/saved_models"
    cuda = 0
    seed = 42
    content_weight = 1e5
    style_weight = 1e10
    lr = 1e-3
    log_interval = 500
    checkpoint_interval = 2000
    content_scale = 1.0

    #content_image = str(settings.BASE_DIR) + str(settings.STATIC_URL) + "style_image/content-images/chicago.jpg"
    #output_image = str(settings.BASE_DIR) + str(settings.STATIC_URL) + "style_image/output-images/test2.jpg"
    output_dir =  str(settings.BASE_DIR) + str(settings.MEDIA_URL)+"images/{:04}".format(pk)
    output_image = output_dir + "/{}.jpg".format(param)
    content_image = str(settings.BASE_DIR) + str(settings.MEDIA_URL)  + str(input_path)

    model = str(settings.BASE_DIR) + str(settings.STATIC_URL) + "style_image/saved_models/"+style+".pth"
    export_onnx = ""  
    image_size = 256
    style_size = 256
    max_style_size = 256
    epochs = 2

    if subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if subcommand == "train":
        check_paths(save_model_dir,checkpoint_model_dir)
        train(cuda,seed,image_size,dataset,batch_size,lr,style_image,style_size,epochs,content_weight,style_weight,log_interval,checkpoint_model_dir,checkpoint_interval,save_model_dir)
    else:
 
        stylize(content_image,content_scale,output_image,model,cuda,export_onnx,max_style_size)


    # c.execute('UPDATE img_trans_imageupload SET "{}" ="{}" WHERE id = "{}";'.format(param,output_path_relative , pk))
    # con.commit()
    # con.close()



# if __name__ == "__main__":
#     main("eval",pk)
    
