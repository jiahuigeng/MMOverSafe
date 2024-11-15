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
import anthropic
import google.generativeai as genai
from utils import load_image

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

def get_claude_model():
    client = anthropic.Anthropic(api_key=claude_api)
    return client

def prompt_claude(client, prompt,  image_id):
    # Initialize the Claude client
    with open(image_id, "rb") as image_file:
        # Encode the image in base64
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    media_type = "image/jpeg"
    # Create a message with the image content
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",  # Specify your model version
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    )

    return message.content[0].text

def get_gemini_model():
    genai.configure(api_key=gemini_api)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

def prompt_gemini(client, prompt, image_id):
    image_data = load_image(image_id)
    response = client.generate_content([image_data, prompt])
    return response.text

if __name__ == "__main__":
    pass
    # client = get_gpt_model()
    # prompt_gpt4(client, "how are you?")

    # claude = get_claude_model()
    # print(prompt_claude(claude, "what's in this image?", "view.jpg"))

    # gemini = get_gemini_model()
    # print(prompt_gemini(gemini, "What is this image?", "view.jpg"))