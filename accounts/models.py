# # Create your models here.
# from django.contrib.auth.models import (AbstractUser, AbstractBaseUser,
#                                         BaseUserManager, PermissionsMixin,
#                                         UserManager)
# from django.db import models

# class BaseModel(models.Model):
#     order = models.IntegerField(null=True, blank=True, verbose_name ='表示順')
#     created = models.DateTimeField(null=True, blank=True, auto_now_add=True, verbose_name ='作成日時')
#     modified = models.DateTimeField(null=True, blank=True, auto_now=True, verbose_name ='更新日時')

#     class Meta:
#         abstract = True

# class User(AbstractUser):

#     class Meta:
#         verbose_name_plural = 'User'
