from django.shortcuts import render, redirect
#from django.contrib.auth.models import User
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.conf import settings
from django.http import HttpResponse
from django.template import loader
from django import forms

import cv2
import sqlite3
import os
import json

# Create your views here.
def login(request):
    
    context = {'temp2':"temp",  }
    print( request.GET.get('username')    )
    print( request.GET.get('email')    )
   
        #user = get_object_or_404(User, pk=user_id)
    print("1--------------------",request.user.username)
    print("2--------------------",request.user)
    print("3--------------------",request.user.is_active)
    print("4--------------------",request.user.is_staff)
    return render(request, 'accounts/login2.html',context)
