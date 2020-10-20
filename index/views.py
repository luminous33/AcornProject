from django.shortcuts import render

# Create your views here.

def main_view(request):
    return render(request, 'index.html')
    
def home(request):
    return render(request, 'home2.html')
    
def new(request):
    full_text = request.GET['fulltext']
    return render(request, 'new.html', { 'fulltext' : full_text })