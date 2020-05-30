from django.urls import path,URLPattern,URLResolver
from django.conf.urls import include, url
from . import views
urlpatterns=[
    path('',views.get_news,name='news'),
    path('news_updates',views.get_news,name='news_updates'),
    path('weather_updates',views.weather_info,name='weather_update'),
    path('weather/',views.weather_info,name='weather'),
]
