from django.shortcuts import render
from django.http import HttpResponse

from .models import CompanyInfo
from .models import DailyPrice

def index(request):
    # companies = CompanyInfo.objects.all()
    # context = {'companies':companies}
        # context에 모든 회사 정보를 저장

    prices = DailyPrice.objects.filter(date="2020-09-29")
    
    context = {'prices':prices}

    return render(request, 'index.html', context)
    
def empty(request):
    return render(request, 'empty.html')
    
def chart(request):
    return render(request, 'chart.html')
    
def introduce(request):
    return render(request, 'introduce.html')
    
def trading(request):
    return render(request, 'trading.html')


