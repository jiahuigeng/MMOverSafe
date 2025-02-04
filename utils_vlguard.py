import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
from builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def get_vlguard_model(model_name):
    if model_name == "vlguard_7b":
        model_path = "ys-zong/llava-v1.5-7b-Mixed"
    elif model_name == "vlguard_13b":
        model_path = "ys-zong/llava-v1.5-13b-Mixed"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=True, device=device)
    return model, tokenizer, image_processor


def get_spavl_model(model_name):
    if model_name == "spavl_ppo":
        model_path = "superjelly/SPA-VL-PPO_30k"
    elif model_name == "spavl_dpo":
        model_path = "superjelly/SPA-VL-DPO_30k"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=True, device=device)
    return model, tokenizer, image_processor

    
# def prompt_vlguard(model, tokenizer, image_processor, prompt, image_file):
#     conv_mode = "llava_v0"
#     conv = conv_templates[conv_mode].copy()
#     roles = conv.roles
    
#     image = load_image(image_file)
#     # Similar operation in model_worker.py
#     # image_tensor = process_images([image], image_processor)
#     image_tensor =  image_processor.preprocess([image], return_tensors='pt')['pixel_values'].to(torch.float16).cuda()
#     if type(image_tensor) is list:
#         image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
#     else:
#         image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        

#     if model.config.mm_use_im_start_end:
#         prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
#     else:
#         prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
#     conv.append_message(conv.roles[0], prompt)
#     image = None

#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()

#     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
#     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#     streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor,
#             do_sample=True,
#             temperature=0.2,
#             max_new_tokens=4096,
#             streamer=streamer,
#             use_cache=True,
#             stopping_criteria=[stopping_criteria])

#     outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
#     # print(outputs)
#     return outputs.replace('</s>', '')
def prompt_vlguard(model, tokenizer, image_processor, prompt, image_file=None):
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    # Initialize variables
    image_tensor = None

    # Process the image only if image_file is provided
    if image_file:
        image = load_image(image_file)
        image_tensor = image_processor.preprocess([image], return_tensors='pt')['pixel_values'].to(torch.float16).cuda()
        if isinstance(image_tensor, list):
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        # Add image tokens to the prompt
        if model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,  # Pass None if no image is used
            do_sample=True,
            temperature=0.2,
            max_new_tokens=4096,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs.replace('</s>', '')

def prompt_spavl(model, tokenizer, image_processor, prompt, image_file=None):
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    # Initialize variables
    image_tensor = None

    # Process the image only if image_file is provided
    if image_file:
        image = load_image(image_file)
        image_tensor = image_processor.preprocess([image], return_tensors='pt')['pixel_values'].to(torch.float16).cuda()
        if isinstance(image_tensor, list):
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        # Add image tokens to the prompt
        if model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,  # Pass None if no image is used
            do_sample=True,
            temperature=0.2,
            max_new_tokens=4096,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs.replace('</s>', '')
    
    
if __name__ == "__main__":
    tokenizer, model, image_processor = get_vlguard_model("vlguard_7b")
    prompt_vlguard(tokenizer, model, image_processor, "what is in this image?", "view.jpg")
    
    prompt_vlguard(tokenizer, model, image_processor, "what is in this image?", None)

    tokenizer, model, image_processor = get_spavl_model("spavl_dpo")
    prompt_vlguard(tokenizer, model, image_processor, "what is in this image?", "view.jpg")
    prompt_vlguard(tokenizer, model, image_processor, "what is in this image?", None)

    tokenizer, model, image_processor = get_spavl_model("spavl_ppo")
    prompt_vlguard(tokenizer, model, image_processor, "what is in this image?", "view.jpg")
    prompt_vlguard(tokenizer, model, image_processor, "what is in this image?", None)