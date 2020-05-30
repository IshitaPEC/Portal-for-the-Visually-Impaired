from __future__ import unicode_literals
from datetime import datetime
from django.db import models

# Create your models here.
class Records(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    name = models.CharField(max_length=50)
    password= models.CharField(max_length= 50)

    def __str__(self):
        return self.name
    class Meta:
        verbose_name_plural = "Records"
