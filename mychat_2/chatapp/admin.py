from django.contrib import admin
from .models import User,Msgs,Job_Details
# Register your models here.
admin.site.register(User)
admin.site.register(Msgs)
admin.site.register(Job_Details)
