from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
from api_resource import *
import torch
import torchvision.transforms as T
from openai import OpenAI
from PIL import Image
from io import BytesIO
import requests
import base64

def get_gpt_model():
    model = OpenAI(api_key=openai_api)
    return model

def prompt_gpt4o(model, prompt, image_id):
    with open(image_id, "rb") as image_file:
        base64_image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image}"
                            # "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    },
                ],
            }
        ],
        temperature=0.01,
        max_tokens=300,
    )
    res = response.choices[0].message.content
    return res

def prompt_gpt4(model, prompt):
    response = model.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.01,
        max_tokens=8192,
    )
    res = response.choices[0].message.content
    return res

if __name__ == "__main__":
    client = get_gpt_model()
    prompt_gpt4(client, "how are you?")