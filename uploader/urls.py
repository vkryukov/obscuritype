from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_csv, name='upload_csv'),
    path('clear/', views.clear_data, name='clear_data'),
] 