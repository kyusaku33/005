from django.shortcuts import render, redirect
#from django.contrib.auth.models import User
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.conf import settings
from django.http import HttpResponse
from django.template import loader
from django import forms
from .forms import SingleUploadModelForm #, ImgForm
from .models import ImageUpload#,Process
import cv2
import sqlite3
import os
import json
from django.contrib.auth.decorators import login_required
import time
from scipy.optimize import fmin_l_bfgs_b
from img_trans.torch.neural_style.neural_style import main


# Create your views here.

#login_required #ログインしてないとログインページにリダイレクト

def top(request):
    form = "temp"
    context = {'form':form,  }

    return render(request, 'img_trans/top.html',context)

def mypage(request):
    EXTRA = 10
    UploadModelFormSet = forms.modelformset_factory(ImageUpload, form=SingleUploadModelForm,extra=EXTRA)
    formset = UploadModelFormSet(request.POST or None, files=request.FILES or None, queryset=ImageUpload.objects.none())
    
    if request.method == 'POST': #画像が選択され，送信されたとき
        if formset.is_valid():

            #アップロードしたレコードにメールアドレスを入力
            list_pk=[]
            for image_temp in  ImageUpload.objects.all():
                list_pk.append(image_temp.pk)
            formset.save()
            list_pk2=[]
            for image_temp in  ImageUpload.objects.all():
                list_pk2.append(image_temp.pk)

            diff_list = set(list_pk) ^ set(list_pk2)
 
            con = sqlite3.connect(str(settings.BASE_DIR)+ '/db.sqlite3')  
            c = con.cursor()
            for diff in  diff_list :
                c.execute('UPDATE img_trans_imageupload SET "Username_id" ="{}" WHERE id = "{}";'.format(request.user.email , diff))
            con.commit()
            con.close()


    #queryset = ImageUpload.objects.filter(Username_id = {request.user.email})#自分のファイルのみ抽出
    queryset_temp =[] 
    queryset = ImageUpload.objects.all()
    
    for query in queryset:
        if query.Username_id == request.user.email:
             queryset_temp.append(query)
    print(queryset_temp)
    paginator = Paginator( queryset_temp, EXTRA )
    p = request.GET.get('page')
    files = paginator.get_page(p)

    context = {
        'files':  files,
        'form': formset,
        'number_list': list(range(EXTRA)),
        'total_number': EXTRA,
    }

    return render(request, 'img_trans/mypage.html',context)

def process(request,pk):
    
    context = {'file': object}
    if request.method == 'POST':
        output_dir =  str(settings.BASE_DIR) + str(settings.MEDIA_URL)+"images/{:04}".format(pk)
        if not os.path.exists(output_dir):    # ディレクトリが存在しない場合、ディレクトリを作成する
            os.makedirs(output_dir)

        image = ImageUpload.objects.get(pk=pk).process(pk) 
        main("eval",pk,"candy","process06")
        main("eval",pk,"mosaic","process07")
        main("eval",pk,"rain_princess","process08")
        main("eval",pk,"udnie","process09")

        # for i in range(10):
        #     print('Start of iteration', i)
        #     start_time = time.time()
        #     x, min_val, info = fmin_l_bfgs_b(ImageUpload.objects.get(pk=pk).evaluator.loss, x.flatten(),fprime=ImageUpload.objects.get(pk=pk).evaluator.grads, maxfun=20)
        #     print('Current loss value:', min_val)
        #     # save current generated image
        #     img = ImageUpload.objects.get(pk=pk).deprocess_image(x.copy())
        #     fname = ImageUpload.objects.get(pk=pk).result_prefix + '_at_iteration_%d.png' % i
        #     ImageUpload.objects.get(pk=pk).save_img(fname, img)
        #     end_time = time.time()
        #     print('Image saved as', fname)
        #     print('Iteration %d completed in %ds' % (i, end_time - start_time))




        return redirect('img_trans:mypage')
    else:
        return render(request, 'img_trans/process.html', context)

  

def delete(request,pk):
    object = ImageUpload.objects.get(pk=pk)
    context = {'file': object}
    if request.method == 'POST':
        object.delete()         
        return redirect('img_trans:mypage')
    else:
        return render(request, 'img_trans/delete.html', context)

###########ここをカスタマイズ############

def gray(input_path,output_path):
    img = cv2.imread(input_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, img_gray)

######################################