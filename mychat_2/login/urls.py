from django.urls import path
from . import views
from django.conf.urls import url
urlpatterns=[
    #path('',views.login,name='login'),
    path('result',views.check,name='check'),
    path('new',views.onetime,name='a'),
    path('',views.create_dataset,name='createdataset'),
    path('eigentrain',views.eigenTrain,name='eigentrain'),
    path('detectimage',views.detectImage,name='detect'),
    path('webcam',views.webcam,name='webcam'),
    #path('assistance',views.lookforassistance,name='ass'),
    #path('listofjobs/',views.listofjobs.as_view(), name="listofjobs"),
    url(r'^details/(?P<id>\w+)/$', views.details, name='details'),
]
