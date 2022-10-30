from django.db import models
import os
import pathlib
import numpy as np
from PIL import Image
import io, base64
from django.conf import settings
import cv2,copy
import sqlite3

class ImageUpload(models.Model):
    files = models.ImageField( upload_to="images", null=True, blank=True, default='')
    #process01 = models.ImageField( upload_to="images", null=True, blank=True, default='')
    #created_at = models.DateTimeField(verbose_name='作成日時', auto_now_add=True)

    # def __str__(self):
    #     self.files
    #     self.process01

    def image_src(self):
        with self.files.open() as img:
            base64_img = base64.b64encode(img.read()).decode()
            return 'data:' + img.file.content_type + ';base64,' + base64_img

    

    def process(self,pk):# 色変換

        img_data = self.files.read()  #
        img_bin = io.BytesIO(img_data)
        image = Image.open(img_bin)
        image = np.array(image, dtype=np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        

        ##加工処理
        image_canny = copy.deepcopy(image)
        image_gray = copy.deepcopy(image)
        img_hsv1 = copy.deepcopy(image)
        img_hsv2 = copy.deepcopy(image)
        img_denoising = copy.deepcopy(image)

        image_canny = cv2.Canny(image, 20, 150)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_hsv1[:, :, 0] = np.where(img_hsv1 [:, :, 0]>300, img_hsv1 [:, :, 0] - 180, img_hsv1[:, :, 0])
        img_hsv2[:, :, 0] = np.where(img_hsv2[:, :, 0]<20, img_hsv2[:, :, 0] + 40, img_hsv2[:, :, 0])
        img_denoising  = cv2.fastNlMeansDenoising(img_denoising , h=20)

        ##データベース登録と保存
        params =["process01","process02","process03","process04","process05",]
        img_paths =[image_canny,image_gray,img_hsv1,img_hsv2,img_denoising ]
        output_dir =  str(settings.BASE_DIR) + str(settings.MEDIA_URL)+"images/{:04}".format(pk)
        con = sqlite3.connect(str(settings.BASE_DIR)+ '/db.sqlite3')  
        c = con.cursor()

        for param,img_path in zip(params, img_paths):
            output_path_relative =  str(settings.MEDIA_URL)+"images/{:04}".format(pk)+ "/"+ param +".jpg"
            cv2.imwrite(output_dir + "/" + param +".jpg" , img_path)  
            c.execute('UPDATE img_trans_imageupload SET "{}" ="{}" WHERE id = "{}";'.format(param,output_path_relative , pk))
        con.commit()
        con.close()
        return  img_bin ,image