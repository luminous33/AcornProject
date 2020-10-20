"""Investar URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from hello import views
from django.urls import path, re_path, include
#from index.views import main_view, home, new 
from balance import views as balance_views
from stock.views import homepage, empty, chart, introduce, trading, bol2, triple, search

urlpatterns = [
    path('admin/', admin.site.urls),
    #re_path(r'^(?P<name>[A-Z][a-z]*)$', views.sayHello),
    #path('index/', main_view),
    #path('indextest/', home),
    #path('new/',new),
    path('balance/', balance_views.main_view),
    path('test1/',include('test1.urls')),
    
    #------------------------
    #path('homepage/',include('homepage.urls')),
    path('homepage/', homepage),
    path('empty', empty),
    path('chart', chart),
    path('introduce', introduce),
    path('trading', trading),
    path('bol2',bol2),
    path('triple',triple),
    path('search',search),
]
