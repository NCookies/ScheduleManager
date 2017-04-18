# chatting/chatbot/urls.py

from django.conf.urls import include, url
from .views import FBChatBotView

urlpatterns = [
                  url(r'^8b95082359388efe22755c24950c009ed29c60ed761ffa04e2/?$', FBChatBotView.as_view())
               ]