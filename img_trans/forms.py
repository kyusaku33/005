from django import forms
from .models import ImageUpload
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class SingleUploadModelForm(forms.ModelForm):
    class Meta:
        model = ImageUpload
        fields = '__all__'

# class ImgForm(forms.Form):
#     photos_field = forms.ImageField( widget=forms.ClearableFileInput(attrs={'multiple': True}))
