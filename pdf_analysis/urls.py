from django.urls import path
from .views import upload_pdf
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', upload_pdf, name='upload_pdf'),
    # path("chatbot/", chatbot_view, name="chatbot"),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)