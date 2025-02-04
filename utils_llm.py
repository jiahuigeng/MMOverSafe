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

def prompt_claude(client, prompt, image_path):

    try:
        # Open the image
        with Image.open(image_path) as img:
            # Check and convert unsupported formats
            format_to_mime = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/jpeg",  # Convert WEBP to JPEG
            }

            image_format = img.format.upper()
            if image_format not in format_to_mime:
                raise ValueError(f"Unsupported image format: {image_format}")

            # Convert WEBP to JPEG if needed
            if image_format == "WEBP":
                img = img.convert("RGB")  # Convert WEBP to RGB for JPEG compatibility
                image_format = "JPEG"  # Update format for saving

            media_type = format_to_mime[image_format]

            # Save the image to a buffer and encode it in Base64
            buffered = io.BytesIO()
            img.save(buffered, format=image_format)
            image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        raise ValueError(f"Failed to process the image: {e}")

    # Construct the API message
    try:
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
        time.sleep(1)
        return message.content[0].text

    except Exception as e:
        raise RuntimeError(f"Failed to query the Claude API: {e}")

def get_gemini_model():
    genai.configure(api_key=gemini_api)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

def prompt_gemini(client, prompt, image_id):
    image_data = load_image(image_id)
    response = client.generate_content([image_data, prompt])
    return response.text



import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import requests
from utils import load_image
import torch

from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

def get_deepseek_model(model_name):
    model_path = "deepseek-ai/deepseek-vl-7b-chat"

    # Load the VLChatProcessor and tokenizer
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Load the MultiModalityCausalLM model
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    return vl_gpt, tokenizer, vl_chat_processor 


# def prompt_deepseek(model, tokenizer, image_processor, prompt, image_file):
#     conversation = [
#     {
#         "role": "User",
#         "content": f"<image_placeholder>{prompt}",
#         "images": [image_file]  # Replace with your image path
#     },
#     {
#         "role": "Assistant",
#         "content": ""
#     }
#     ]

#     # Load images and prepare for inputs
#     pil_images = load_pil_images(conversation)

#     # Prepare inputs for the model
#     prepare_inputs = image_processor(
#         conversations=conversation,
#         images=pil_images,
#         force_batchify=True
#     ).to(model.device)

#     # Run image encoder to get the image embeddings
#     inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

#     # Generate a response from the model
#     outputs = model.language_model.generate(
#         inputs_embeds=inputs_embeds,
#         attention_mask=prepare_inputs.attention_mask,
#         pad_token_id=tokenizer.eos_token_id,
#         max_new_tokens=512,
#         do_sample=False,
#         use_cache=True
#     )

#     answer = tokenizer.decode(outputs.cpu().tolist()[0], skip_special_tokens=True)

#     return answer

def prompt_deepseek(model, tokenizer, image_processor, prompt, image_file=None):
    """
    Generates a response from the DeepSeek model using optional image input.

    Args:
        model: The DeepSeek model instance.
        tokenizer: Tokenizer for the model.
        image_processor: Image processor to prepare image data.
        prompt (str): The text prompt for the model.
        image_file (str, optional): The file path of the image. Defaults to None.

    Returns:
        str: The model's response text.
    """
    try:
        # Initialize conversation
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
                "images": [image_file] if image_file else []
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        if image_file:
            # Load images and prepare inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = image_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(model.device)

            # Run image encoder to get the image embeddings
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

            # Generate a response from the model
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )
        else:
            # Generate a response without image input
            outputs = model.language_model.generate(
                input_ids=tokenizer(prompt, return_tensors="pt").input_ids.to(model.device),
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True
            )

        # Decode and return the response
        answer = tokenizer.decode(outputs.cpu().tolist()[0], skip_special_tokens=True)
        return answer

    except Exception as e:
        # raise RuntimeError(f"An error occurred while querying the DeepSeek model: {e}")
        print(f"Error {e}")
        return ""
    

def preprocess_image(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Define transformations
    transform = T.Compose([
        T.Resize((448, 448)),  # Resize to the expected input size
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    # Apply transformations, add batch dimension, and convert to bfloat16
    return transform(image).unsqueeze(0).to(torch.bfloat16).cuda()


def get_internvl_model(model_name):
    
    processor = None
    if model_name == "internvl_4b":
        full_model_name = "OpenGVLab/InternVL2-4B" 
    if model_name == "internvl_8b":
        full_model_name = "OpenGVLab/InternVL2-8B"
    elif model_name == "internvl_26b":
        full_model_name = "OpenGVLab/InternVL2-26B"
        
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
    full_model_name,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
    
    return model, tokenizer, processor

# def prompt_internvl(model, tokenizer, processor, prompt, image_id):
#     image_tensor = preprocess_image(image_id)
#     generation_config = {
#     "num_beams": 1,
#     "max_new_tokens": 4096,
#     "do_sample": False,
#     }
#     question = f"<image>\n{prompt}"
#     # print(f"question: {question}")
#     response = model.chat(tokenizer, pixel_values=image_tensor, question=question, generation_config=generation_config)
#     return response

def prompt_internvl(model, tokenizer, processor, prompt, image_path=None):
    """
    Generates a response from the InternVL model using optional image input.

    Args:
        model: The InternVL model instance.
        tokenizer: Tokenizer for the model.
        processor: Processor to preprocess image data.
        prompt (str): The text prompt for the model.
        image_path (str, optional): The file path of the image. Defaults to None.

    Returns:
        str: The model's response text.
    """
    try:
        generation_config = {
            "num_beams": 1,
            "max_new_tokens": 4096,
            "do_sample": False,
        }

        if image_path:
            # Preprocess the image if provided
            image_tensor = preprocess_image(image_path)
            question = f"<image>\n{prompt}"

            # Generate response using image and text
            response = model.chat(
                tokenizer,
                pixel_values=image_tensor,
                question=question,
                generation_config=generation_config,
            )
        else:
            # Generate response using only the text prompt
            question = prompt
            response = model.chat(
                tokenizer,
                pixel_values=None,
                question=question,
                generation_config=generation_config,
            )

        return response

    except Exception as e:
        # raise RuntimeError(f"An error occurred while querying the InternVL model: {e}")
        print(f"Error {e}")
        return ""


# def get_llava_model(model_name):
#     if model_name == "llava_7b":
#         model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
#     elif model_name == "llava_13b":
#         model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
        
#     processor = LlavaNextProcessor.from_pretrained(model_id)
#     model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True) 
#     model.to("cuda")
    
#     tokenizer = None
#     return model, tokenizer, processor


# def prompt_llava(model, tokenizer, processor, prompt, image_id):
#     conversation = [
#         {

#             "role": "user",
#             "content": [
#                 {"type": "text", f"text": f"{prompt}"},
#                 {"type": "image"},
#             ],
#         },
#     ]
#     image = load_image(image_id)
#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

#     inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

#     # autoregressively complete prompt
#     output = model.generate(**inputs, max_new_tokens=4096)

#     return processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

def get_llava_model(model_name):
    if model_name == "llava_7b":
        model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
    elif model_name == "llava_13b":
        model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True
    )
    model.to("cuda")
    
    tokenizer = None  # Update if a tokenizer is needed later
    return model, tokenizer, processor


# def prompt_llava(model, tokenizer, processor, prompt, image_path=None):
#     try:
#         # Create the conversation template
#         conversation = [
#             {
#                 "role": "user",
#                 "content": [{"type": "text", "text": prompt}],
#             },
#         ]

#         inputs = None  # Initialize inputs to None
#         # Include image in the conversation if provided
#         if image_path:
#             image = load_image(image_path)
#             if image is None:
#                 raise ValueError("Failed to load image. Ensure the image path is valid.")
            
#             conversation[0]["content"].append({"type": "image"})
#             inputs = processor(
#                 images=image,
#                 text=processor.apply_chat_template(conversation, add_generation_prompt=True),
#                 return_tensors="pt"
#             )
#         else:
#             inputs = processor(
#                 text=#processor.apply_chat_template(conversation, add_generation_prompt=True),
#                 return_tensors="pt"
#             )

#         # Check if inputs are valid
#         if inputs is None:
#             raise ValueError("Processor returned None. Ensure the inputs are correctly configured.")
        
#         # Move inputs to the appropriate device
#         inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

#         # Autoregressively complete the prompt
#         output = model.generate(**inputs, max_new_tokens=4096)

#         # Decode and format the output
#         return processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

#     except Exception as e:
#         # Log the error and return an empty string
#         print(f"Error: {e}")
#         return ""

def prompt_llava(model, tokenizer, processor, prompt, image_path=None):
    try:
        # Create the conversation template
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

        inputs = None  # Initialize inputs to None
        # Include image in the conversation if provided
        if image_path:
            image = load_image(image_path)
            if image is None:
                raise ValueError("Failed to load image. Ensure the image path is valid.")
            
            conversation[0]["content"].append({"type": "image"})
            inputs = processor(
                images=image,
                text=processor.apply_chat_template(conversation, add_generation_prompt=True),
                return_tensors="pt"
            )
            
            # # Check if inputs are valid
            if inputs is None:
                raise ValueError("Processor returned None. Ensure the inputs are correctly configured.")
            

            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

            # Autoregressively complete the prompt
            output = model.generate(**inputs, max_new_tokens=4096)

            # Decode and format the output
            return processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
        else:
            inputs = processor(text=prompt, return_tensors="pt")
            inputs = {k: v.to("cuda:0") if hasattr(v, 'to') else v 
                    for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
                
            
            response = processor.decode(outputs[0], skip_special_tokens=True)
            return response



    except Exception as e:
        # Log the error and return an empty string
        print(f"Error: {e}")
        return ""

    
def get_commercial_model(model_name):
    model_map = {
        "gpt4o": get_gpt_model,
        "claude": get_claude_model,
        "gemini": get_gemini_model,
    }
    if model_name in model_map:
        return model_map[model_name]()
    raise ValueError(f"Unknown model name: {model_name}")

def prompt_commercial_model(client, model_name, prompt, image_id):
    prompt_map = {
        "gpt4o": prompt_gpt4o,
        "claude": prompt_claude,
        "gemini": prompt_gemini,
    }
    if model_name in prompt_map:
        try:
            return prompt_map[model_name](client, prompt, image_id)
        except Exception as e:
            print(image_id, str(e))
            return ""
    raise ValueError(f"Unknown model name: {model_name}")

def get_open_model(model_name):
    model_map = {
        "llava_7b": get_llava_model,
        "llava_13b": get_llava_model,
        "internvl_8b": get_internvl_model,
        "deepseek": get_deepseek_model,
    }
    if model_name in model_map:
        return model_map[model_name](model_name)
    raise ValueError(f"Unknown model name: {model_name}")
    
def prompt_open_model(model, tokenizer, processor, model_name, prompt, image_id=None):
    prompt_map = {
        "llava_7b": prompt_llava,
        "llava_13b": prompt_llava,
        "internvl_8b": prompt_internvl,
        "deepseek": prompt_deepseek,
    }
    if model_name in prompt_map:
        try:
            return prompt_map[model_name](model, tokenizer, processor, prompt, image_id)
        except Exception as e:
            print(image_id, str(e))
            return ""
    raise ValueError(f"Unknown model name: {model_name}")

if __name__ == "__main__":
    
    # pass
    # model, tokenizer, processor = get_deepseek_model("deepseek")
    # resp = prompt_deepseek(model, tokenizer, processor, "what is in this image?", "view.jpg")
    # print(resp)
    
    # model, tokenizer, processor = get_deepseek_model("deepseek")
    # resp = prompt_deepseek(model, tokenizer, processor, "what is in this image?")
    # print(resp)
    
    # model, tokenizer, processor = get_internvl_model("internvl_8b")
    # resp = prompt_internvl(model, tokenizer, processor, "what is in this image?", "view.jpg")
    # print(resp)
    
    # model, tokenizer, processor = get_internvl_model("internvl_8b")
    # resp = prompt_internvl(model, tokenizer, processor, "what is in this image?")
    # print(resp)
    
    
    # model, tokenizer, processor = get_internvl_model("internvl_4b")
    # resp = prompt_internvl(model, tokenizer, processor, "what is in this image?", "butterfly.jpg")
    # print(resp)
    
    model, tokenizer, processor = get_llava_model("llava_7b")
    prompt = "What is shown in this image?"
    image_id = "view.jpg"
    response = prompt_llava(model, tokenizer, processor, prompt, image_id)
    print(response)
    
    response = prompt_llava(model, tokenizer, processor, prompt, None)
    print(response)
    
    # model, tokenizer, processor = get_llava_model("llava_13b")
    # prompt = "What is shown in this image?"
    # image_id = "view.jpg"
    # response = prompt_llava(model, tokenizer, processor, prompt, image_id)
    # print(response)
    
    # client = get_gpt_model()
    # prompt_gpt4(client, "how are you?")

    # claude = get_claude_model()
    # print(prompt_claude(claude, "what's in this image?", "view.jpg"))

    # gemini = get_gemini_model()
    # print(prompt_gemini(gemini, "What is this image?", "view.jpg"))