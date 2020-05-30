from django.db import models

# Create your models here.
class User(models.Model):
    name=models.CharField(max_length=20)
    pswd=models.CharField(max_length=20)
    msg=models.TextField()

class Msgs(models.Model):
    username=models.CharField(max_length=20)
    receiver=models.CharField(max_length=20)
    msg=models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    del_sen = models.BooleanField(default=False)
    del_rec = models.BooleanField(default=False)
    read = models.BooleanField(default=False)

class Job_Details(models.Model):
    job_name=models.CharField(max_length=100)
    company_name=models.CharField(max_length=100)
    location=models.CharField(max_length=100)
    salary=models.CharField(max_length=100)
    url= models.URLField(max_length=1000)

    def __str__(self):
        return self.job_name
