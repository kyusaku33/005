# Generated by Django 4.1.2 on 2022-10-23 05:43

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('img_trans', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='imageupload',
            name='created_at',
        ),
    ]