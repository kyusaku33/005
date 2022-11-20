from django.db import models
import os
import pathlib
import numpy as np
from PIL import Image
import io, base64
from django.conf import settings
import cv2,copy
import sqlite3
from django.contrib.auth.models import User
from img_trans.image import changedH

from img_trans.torch.neural_style.utils import load_image,load_image_style,save_image,gram_matrix,normalize_batch
#from .model_cnn import preprocess_image
#from .model_fast_style import main

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
#from __future__ import print_function
# from keras.preprocessing.image import load_img, save_img, img_to_array
# import numpy as np
# from scipy.optimize import fmin_l_bfgs_b
# import time

# from keras.applications import vgg19
# from keras import backend as K


class ImageUpload(models.Model):
    files = models.ImageField( upload_to="images", null=True, blank=True, default='')
    Username = models.ForeignKey(User,on_delete=models.CASCADE, null=True,blank=True)
    process01 = models.CharField(verbose_name="process01",blank=True,null=True,default='',max_length=2000)
    process02 = models.CharField(verbose_name="process02",blank=True,null=True,default='',max_length=2000)
    process03 = models.CharField(verbose_name="process03",blank=True,null=True,default='',max_length=2000)
    process04 = models.CharField(verbose_name="process04",blank=True,null=True,default='',max_length=2000)
    process05 = models.CharField(verbose_name="process05",blank=True,null=True,default='',max_length=2000)
    process06 = models.CharField(verbose_name="process06",blank=True,null=True,default='',max_length=2000)
    process07 = models.CharField(verbose_name="process07",blank=True,null=True,default='',max_length=2000)
    process08 = models.CharField(verbose_name="process08",blank=True,null=True,default='',max_length=2000)
    process09 = models.CharField(verbose_name="process09",blank=True,null=True,default='',max_length=2000)
    process10 = models.CharField(verbose_name="process10",blank=True,null=True,default='',max_length=2000)
    process11 = models.CharField(verbose_name="process11",blank=True,null=True,default='',max_length=2000)
    process12 = models.CharField(verbose_name="process12",blank=True,null=True,default='',max_length=2000)
    #process01 = models.ImageField( upload_to="images", null=True, blank=True, default='')
    #created_at = models.DateTimeField(verbose_name='作成日時', auto_now_add=True)

    


    def image_src(self):
        with self.files.open() as img:
            base64_img = base64.b64encode(img.read()).decode()
            
            return 'data:' + img.file.content_type + ';base64,' + base64_img


    def process(self,pk):# 色変換
        
        # img_data = self.files.read()  #
        # img_bin = io.BytesIO(img_data)
        # image = Image.open(img_bin)
        # image = np.array(image, dtype=np.uint8)
        #モノクロの場合カラーへ
        #image2 = Image.open(self.files)
        
        #image = load_image_style(self.files, scale=1.0, max_style_size=1024)
        content_image = str(settings.BASE_DIR) + str(settings.MEDIA_URL)  + self.files.name

        #image = cv2.imread( content_image)
        image = load_image_style(content_image, scale=1.0, max_style_size=1024)
        output_dir =  str(settings.BASE_DIR) + str(settings.MEDIA_URL)+"images/{:04}".format(pk)
        
        #image2.save(output_dir + "/"  +"21.jpg")
        image = np.array(image, dtype=np.uint8)

        cv2.imwrite(output_dir + "/"  +"22.jpg" , image) 
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #画像サイズ調整
        # w, h = image.shape[:2]
        # print(h,w)
        # height = round((h/ w) * 1024 )
        # if w > 1024:
        #     image = cv2.resize(image, dsize=(1024, height))
        #     output_dir =  str(settings.BASE_DIR) + str(settings.MEDIA_URL)+"images/{:04}".format(pk)
        #     cv2.imwrite(output_dir + "/"  +"22.jpg" , image) 
        
        ##加工処理
        # image_canny = copy.deepcopy(image)
        # image_gray = copy.deepcopy(image)
        # img_hsv1 = copy.deepcopy(image)
        # img_hsv2 = copy.deepcopy(image)
        # img_denoising = copy.deepcopy(image)
        image_canny = cv2.Canny(image, 20, 150)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_hsv1 = changedH(image, 85)
        img_hsv2 = changedH(image, -85)

        img_denoising  = cv2.fastNlMeansDenoising(image_gray , h=20)

        ##データベース登録と保存
        params =["process01","process02","process03","process04","process05",]
        img_paths =[image_canny,image_gray,img_hsv1,img_hsv2,img_denoising ]
        #img_paths =[image_canny,image_canny,image_canny,image_canny,image_canny ]
        
        output_dir =  str(settings.BASE_DIR) + str(settings.MEDIA_URL)+"images/{:04}".format(pk)
        con = sqlite3.connect(str(settings.BASE_DIR)+ '/db.sqlite3')  
        c = con.cursor()

        for param,img_path in zip(params, img_paths):
            output_path_relative =  str(settings.MEDIA_URL)+"images/{:04}".format(pk)+ "/"+ param +".jpg"
            cv2.imwrite(output_dir + "/" + param +".jpg" , img_path)  
            c.execute('UPDATE img_trans_imageupload SET "{}" ="{}" WHERE id = "{}";'.format(param,output_path_relative , pk))
        con.commit()
        con.close()
        return   image

    # def process02(self,pk):# 色変換

    #     img_data = self.files.read()  #
    #     img_bin = io.BytesIO(img_data)
    #     image = Image.open(img_bin)
    #     image = np.array(image, dtype=np.uint8)
    #     if len(image.shape) == 2:
    #         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)

      
        