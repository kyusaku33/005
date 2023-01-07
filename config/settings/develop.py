from .base import * # base.pyを読み込む 

# 開発、本番で分けたい設定を記載
# ssh ec2-user@35.78.189.235 -i C:\Users\masud\Downloads\test-key-1.pem -L 15432:database-1.crr3xmeryalf.ap-northeast-1.rds.amazonaws.com:5432
# ssh ec2-user@35.78.189.235 -i C:\Users\masud\Downloads\test-key-1.pem
# https://github.com/masudanaokibi/django2.git

# DEBUG = True

# ALLOWED_HOSTS = []

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql',
#         'NAME': 'djangodb3',
#         'USER': 'postgres',
#         
#         'HOST': 'localhost',
#         'PORT': '15432'
#     }
# }
