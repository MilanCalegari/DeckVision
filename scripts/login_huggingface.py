import os

from huggingface_hub import login

token = os.getenv('HUGGINGFACE_TOKEN')
if token:
    login(token=token)