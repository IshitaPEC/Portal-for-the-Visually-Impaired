# Generated by Django 2.2.1 on 2020-04-27 14:11

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Job_Details',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('job_name', models.CharField(max_length=100)),
                ('company_name', models.CharField(max_length=100)),
                ('location', models.CharField(max_length=100)),
                ('salary', models.CharField(max_length=100)),
                ('summary', models.CharField(max_length=1000)),
            ],
        ),
        migrations.CreateModel(
            name='Records',
            fields=[
                ('id', models.CharField(max_length=100, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=50)),
                ('password', models.CharField(max_length=50)),
            ],
            options={
                'verbose_name_plural': 'Records',
            },
        ),
    ]
