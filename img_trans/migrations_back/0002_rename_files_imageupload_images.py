# Generated by Django 4.1.2 on 2022-10-23 03:30

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('img_trans', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='imageupload',
            old_name='files',
            new_name='images',
        ),
    ]
