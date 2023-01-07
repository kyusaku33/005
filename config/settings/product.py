from .base import * # base.pyを読み込む 

# 開発、本番で分けたい設定を記載1

#DEBUG = False

ALLOWED_HOSTS = ["*"]


DATABASES = {
    'default': {
         'ENGINE': 'django.db.backends.postgresql_psycopg2',
        #'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'postgres',
        'USER': 'postgres',
        'PASSWORD': 'F12000aws',
        'HOST': 'postgres-005.chjbf4ut1prw.ap-northeast-1.rds.amazonaws.com',
        'PORT': '5432'
    }
}


STATIC_ROOT = '/usr/share/nginx/html/static'
MEDIA_ROOT = '/usr/share/nginx/html/media'