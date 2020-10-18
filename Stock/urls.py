from django.contrib import admin
from django.urls import path,include
from . import views


urlpatterns = [
    path('',views.homepage,name="home"),
    path('homepage/',views.homepage,name="home"),
    path('bol/',views.bol,name="home"),
]