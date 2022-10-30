"""
Django settings for config project.

Generated by 'django-admin startproject' using Django 4.1.2.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.1/ref/settings/
"""

from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
import os

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-4#6^o+=$)u6%o4fr!mhzs)mj2cmn!l!)6cf!fdruba#dr-+ew('

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'django_static_md5url',# 追加
    'widget_tweaks',# 追加

    'django.contrib.sites', # 追加
    'allauth', # 追加
    'allauth.account', # 追加
    'allauth.socialaccount', # 追加
    #'user_sessions', #追加

    'accounts.apps.AccountsConfig',
    'img_trans', # 追加
]

SESSION_ENGINE = 'django.contrib.sessions.backends.cache'# 追加
#SESSION_ENGINE = 'user_sessions.backends.db'# 追加

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
   # 'user_sessions.middleware.SessionMiddleware',  #追加
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
      'DIRS': [
      os.path.join(BASE_DIR, 'templates'),
      #os.path.join(BASE_DIR, 'templates', 'accounts'),
      os.path.join(BASE_DIR, 'accounts','templates'),
      #os.path.join(BASE_DIR, 'templates', 'custom_auth'),
      ],

        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
       # 'ATOMIC_REQUESTS': True,  # 追加
    }
}


# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = 'ja'

TIME_ZONE = 'Asia/Tokyo'

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.1/howto/static-files/

STATIC_URL = 'static/'
STATICFILES_DIRS = [ BASE_DIR / 'static']

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

#STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.ManifestStaticFilesStorage'
# Default primary key field type
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


#AUTH_USER_MODEL = 'accounts.User'

# django-allauthで利用するdjango.contrib.sitesを使うためにサイト識別用IDを設定
SITE_ID = 1

AUTHENTICATION_BACKENDS = (
    'allauth.account.auth_backends.AuthenticationBackend',  # 一般ユーザー用(メールアドレス認証)
    'django.contrib.auth.backends.ModelBackend',  # 管理サイト用(ユーザー名認証)
)

# メールアドレス認証に変更する設定
ACCOUNT_AUTHENTICATION_METHOD = 'email'
ACCOUNT_USERNAME_REQUIRED = False

# サインアップにメールアドレス確認をはさむよう設定
ACCOUNT_EMAIL_VERIFICATION = 'mandatory'
ACCOUNT_EMAIL_REQUIRED = True

# ログイン/ログアウト後の遷移先を設定
#LOGIN_URL = 'img_trans:top'
LOGIN_REDIRECT_URL = 'img_trans:top'
ACCOUNT_LOGOUT_REDIRECT_URL = 'account_login'

# ログアウトリンクのクリック一発でログアウトする設定
ACCOUNT_LOGOUT_ON_GET = True

# django-allauthが送信するメールの件名に自動付与される接頭辞をブランクにする設定
ACCOUNT_EMAIL_SUBJECT_PREFIX = ''

# デフォルトのメール送信元を設定
# DEFAULT_FROM_EMAIL = os.environ.get('FROM_EMAIL')

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_USE_TLS = True
EMAIL_PORT = 587
EMAIL_HOST_USER = 'kyusakunsqc@gmail.com'
EMAIL_HOST_PASSWORD = 'jcgliwfqthvuarvd'


ACCOUNT_FORMS = {
    'signup': 'accounts.forms.CustomSignupForm',
    'login': 'accounts.forms.CustomLoginForm',
    'reset_password': 'accounts.forms.CustomResetPasswordForm',
    'reset_password_from_key': 'accounts.forms.CustomResetPasswordKeyForm',
    'change_password': 'accounts.forms.CustomChangePasswordForm',
    'add_email': 'accounts.forms.CustomAddEmailForm',
    'set_password': 'accounts.forms.CustomSetPasswordForm',
}



# AUTH_USER_MODEL = 'accounts.User'
# #AUTH_USER_MODEL = 'accounts.CustomUser'


# # ACCOUNT_EMAIL_REQUIRED = True # Eメール必須
# # ACCOUNT_USERNAME_REQUIRED = True # ユーザー名必須
# # ACCOUNT_SIGNUP_PASSWORD_ENTER_TWICE = True # パスワード２回入力
# # ACCOUNT_SESSION_REMEMBER = True
# # ACCOUNT_AUTHENTICATION_METHOD = 'email' # Eメールで認証を行う
# # ACCOUNT_UNIQUE_EMAIL = True # 一意なEメール
# # ACCOUNT_EMAIL_VERIFICATION = 'mandatory' # Eメール送信



# #AUTH_USER_MODEL = 'users.User'