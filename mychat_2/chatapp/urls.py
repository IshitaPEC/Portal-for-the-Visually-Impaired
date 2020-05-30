from django.urls import path,URLPattern,URLResolver
from django.conf.urls import include, url
from . import views
urlpatterns=[
    path('',views.home,name='home'),
    path('intermediate',views.inter,name='intermediate'),
    path('result',views.check,name='result'),
    path('new', views.new, name='new'),
    path('assistance',views.lookforassistance,name='ass'),
    path('listofjobs',views.listofjobs,name='listofjobs'),
    url(r'^chatbase/(?P<receiver>\w+)/$', views.chatbase, name='chatbase'),
    url(r'^chatbase/(?P<receiver>\w+)/writemessage/$', views.send_msg, name='writemessage'),
    url(r'^chatbase/(?P<receiver>\w+)/listenmessage/$', views.read_msgs, name='listenmessage'),
]
