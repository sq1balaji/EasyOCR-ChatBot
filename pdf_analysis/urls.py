from django.urls import path
from . import views

urlpatterns = [
    # URL for uploading a PDF
    path('upload/', views.upload_pdf, name='upload_pdf'),
    
    # URL for the chatbot interaction
    path('chatbot/', views.chatbot, name='chatbot'),

    # URL for displaying the results of the PDF analysis
    path('results/', views.upload_pdf, name='results'),  # Update this with correct view if different

    # URL for other functionalities if needed
    path('', views.upload_pdf, name='home'),  # Default home page route
]
