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
from django.forms import BaseModelFormSet

# Create your views here.

#login_required #ログインしてないとログインページにリダイレクト

# class MyModelFormSet(BaseModelFormSet):
#     def clean(self):
#         super().clean()

#         for form in self.forms:
#             name = form.cleaned_data['name'].upper()
#             form.cleaned_data['Username'] = name
#             # update the instance value.
#             form.instance.name = name


def top(request):
    form = "temp"
    context = {'form':form,  }

    return render(request, 'img_trans/top.html',context)

def mypage(request):
    EXTRA = 1
    UploadModelFormSet = forms.modelformset_factory(ImageUpload, form=SingleUploadModelForm,extra=EXTRA)
    formset = UploadModelFormSet(request.POST or None, files=request.FILES or None, queryset=ImageUpload.objects.none())
    
    if request.method == 'POST': #画像が選択され，送信されたとき
        if formset.is_valid():
            for data in formset:
                 test = data.save(commit=False)
                 #test.email = request.user.email 
                 test.Username = request.user
                 test.save()
               
            formset.save()
    


    queryset = ImageUpload.objects.filter(Username = request.user)#自分のファイルのみ抽出
    #queryset_temp =[] 
    #queryset = ImageUpload.objects.all()
    
    #for query in queryset:
    #    if query.email == request.user.email:
    #         queryset_temp.append(query)
 
    #print(queryset_temp)
    # paginator = Paginator( queryset_temp, EXTRA )
    paginator = Paginator( queryset, 20 )

    p = request.GET.get('page')
    files = paginator.get_page(p)

    context = {
        'files':  files,
        'form': formset,
        'number_list': list(range(20)),
        'total_number': 20,
    }

    return render(request, 'img_trans/mypage.html',context)

def process(request,pk):
    
    context = {'file': object}
    if request.method == 'POST':
        output_dir =  str(settings.BASE_DIR) + str(settings.MEDIA_URL)+"images/{:04}".format(pk)
        if not os.path.exists(output_dir):    # ディレクトリが存在しない場合、ディレクトリを作成する
            os.makedirs(output_dir)

        image = ImageUpload.objects.get(pk=pk).process(pk)
        db = ImageUpload.objects.get(pk=pk) 

        main("eval",pk,"candy","process06",db.files)
        output_path_relative =  str(settings.MEDIA_URL)+"images/{:04}".format(pk)+ "/" +"process06.jpg"
        db.process06 =  output_path_relative
        main("eval",pk,"mosaic","process07",db.files)
        output_path_relative =  str(settings.MEDIA_URL)+"images/{:04}".format(pk)+ "/" +"process07.jpg"
        db.process07 =  output_path_relative
        main("eval",pk,"rain_princess","process09",db.files)
        output_path_relative =  str(settings.MEDIA_URL)+"images/{:04}".format(pk)+ "/" +"process08.jpg"
        db.process08 =  output_path_relative
        main("eval",pk,"udnie","process09",db.files)
        output_path_relative =  str(settings.MEDIA_URL)+"images/{:04}".format(pk)+ "/" +"process09.jpg"
        db.process09 =  output_path_relative



        # main("eval",pk,"mosaic","process07")
        # main("eval",pk,"rain_princess","process08")
        # main("eval",pk,"udnie","process09")
        db.save()
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