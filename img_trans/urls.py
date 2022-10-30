from django.urls import path

from . import views
#from views import views, upload, webapi

app_name = 'img_trans'

urlpatterns = [

    path("", views.top, name='top'),
    path("mypage/", views.mypage, name='mypage'),
    path('delete/<int:pk>/', views.delete, name='delete'),
    path('/<int:pk>/', views.process, name='process'),
    
    #path('delete/<int:pk>/', views.delete, name='delete'),

]


