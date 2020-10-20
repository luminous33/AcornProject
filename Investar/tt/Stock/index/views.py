from django.shortcuts import render

def index(request):
    return render(request, 'index.html')
    
def empty(request):
    return render(request, 'empty.html')
    
def chart(request):
    return render(request, 'chart.html')
    
def introduce(request):
    return render(request, 'introduce.html')
    
def trading(request):
    return render(request, 'trading.html')
