from huggingface_hub import HfApi
from config import HF_TOKEN

api = HfApi()
user = api.whoami(token=HF_TOKEN)
print(f"Authenticated as: {user['name']}")