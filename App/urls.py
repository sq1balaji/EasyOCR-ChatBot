# In your_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('chatbot/', views.chatbot, name='chatbot'),  # Ensure this line exists
]