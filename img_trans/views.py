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

# Create your views here.
def top(request):
    form = "temp"
    context = {'form':form,  }

    return render(request, 'img_trans/top.html',context)

def mypage(request):
    EXTRA = 10
    UploadModelFormSet = forms.modelformset_factory(ImageUpload, form=SingleUploadModelForm,extra=EXTRA)
    formset = UploadModelFormSet(request.POST or None, files=request.FILES or None, queryset=ImageUpload.objects.none())
    
    if request.method == 'POST': #画像が選択され，送信されたとき
        formset.save() #データベースに保存

    queryset = ImageUpload.objects.all()
    paginator = Paginator( queryset, EXTRA )
    p = request.GET.get('page')
    files = paginator.get_page(p)

    con = sqlite3.connect(str(settings.BASE_DIR)+ '/db.sqlite3')  
    c = con.cursor()
    c.execute("select * from img_trans_imageupload")
    i = 0
    temp =c.fetchall()

    #c.execute("alter table img_trans_imageupload add column process01") #最初だけ
    #c.execute('UPDATE img_trans_imageupload SET process01 ="{}" WHERE id = "{}";'.format(output_path_relative , pk))
    con.commit()
    con.close()

    context = {
        'zip':  zip(temp, files),
        # 'queryset': temp,
        # 'files': files,
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

        #predicted01, rate01 = ImageUpload.objects.get(pk=pk).process01(pk)  
        predicted, rate = ImageUpload.objects.get(pk=pk).process(pk) 
        return redirect('img_trans:mypage')
    else:
        return render(request, 'img_trans/process.html', context)

    # if not request.method == 'POST':
    #     return redirect('img_trans:mypage')
    
    # form = SingleUploadModelForm(request.POST, request.FILES) #フォーム呼び出し
    # if not form.is_valid():
    #     raise ValueError('Form is illegal.')
    # print(form.cleaned_data['image'])

    # pict = Process(image=form.cleaned_data['image']) #モデル呼び出し
    # print(pict)
    # if 'process01' in request.POST:
    #     # ボタン1がクリックされた場合の処理
    #     predicted, rate = pict.process01()
    #     template = loader.get_template('img_trans:mypage')
    #     context = {
    #     'form2': form ,
    #     'pict_name': pict.image.name,
    #     'pict_data': pict.image_src(),
    #     'predicted': predicted,
    #     'rate': rate,
    #     }
    #     return HttpResponse(template.render(context, request))

    # elif 'process02' in request.POST:
    #     # ボタン2がクリックされた場合の処理
    #     image_org,image,output_path = pict.rectangle()
    #     template = loader.get_template('img_trans:mypage')
    #     #print("---------------",output_path )

    #     img_2 =pict.image_src()

    #     context = {
    #         'pict_name': pict.image.name,
    #         #'pict_data': pict.image_src(), #image_org,W
    #         'pict_out_path': output_path, #image_org,
    #     }

    #     return HttpResponse(template.render(context, request))
    # queryset = ImageUpload.objects.all()
    # context = {
    #     'queryset': queryset,
    # }
    # return render(request, 'img_trans/mypage.html',context)

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