from datasets import load_dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ast
import google.generativeai as genai
import re
import time

from unsloth import FastVisionModel
from transformers import TextStreamer

# model, tokenizer = FastVisionModel.from_pretrained(
#     model_name = "rfeiglew/qwen2vl_refl_7b", # YOUR MODEL YOU USED FOR TRAINING
#     # model_name = "unsloth/gemma-3-4b-it", # YOUR MODEL YOU USED FOR TRAINING
#     load_in_4bit = True, # Set to False for 16bit LoRA
# )
# FastVisionModel.for_inference(model) # Enable for inference!

API_KEY = "<Add your API key here>"
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(model_name='gemini-2.0-flash')


def generate_qwen_vlm_answer(model,tokenizer, image, instruction):
    image = image.resize((480,480))
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=0.1, min_p=0.1)
    generated_answer = tokenizer.decode(outputs[0, len(inputs.input_ids[0]):], skip_special_tokens=True)
    return generated_answer


def generate_gemini_vlm_answer(image, instruction):

    generated_answer = model.generate_content([
        image,
        (
            instruction + "Do not generate ```python``` expressions. Answer in plain text."
        ),
    ])

    generated_answer = generated_answer.text

    # Additional step to remove potential code blocks
    cleaned_answer = re.sub(r"```python\n(.*?)\n```", r"\1", generated_answer, flags=re.DOTALL)
    cleaned_answer = cleaned_answer.replace("```", "") # Remove any remaining ``` markers

    return cleaned_answer

def create_empty_mask(img_width, img_height):
    mask =  np.zeros((img_height,img_width), dtype=np.uint8)
    return mask

def yolo_bbox_to_abs(yolo_box, img_width,img_height):
    class_id, x_center, y_center, width, height = yolo_box

    # convert to absolute values
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    # compute coordinates
    x_min = int(x_center_abs - (width_abs / 2))
    y_min = int(y_center_abs - (height_abs / 2))
    x_max = int(x_center_abs + (width_abs / 2))
    y_max = int(y_center_abs + (height_abs / 2))

    return (class_id, x_min, y_min, x_max, y_max)

def convert_yolo_bboxes_to_abs_bboxes(bboxes,im_width, im_height):
    abs_bboxes = []
    # check if list is not empty
    if  example_bbox:
        for box in bboxes:
            abs_bboxes.append(yolo_bbox_to_abs(box,im_width,im_height))
    return abs_bboxes


def create_mask(boxes, img_width, img_height, excluded_obj=[]):

    # Assure that excluded_obj is a list
    if not isinstance(excluded_obj, list):
        raise TypeError(f"excluded_obj should be a list but is {type(excluded_obj)}: {excluded_obj}")
    

    #initialize empty mask 
    mask = np.zeros((img_width,img_height),dtype=np.uint8)

    if boxes:
        for i, box in enumerate(boxes):  # Dodajemy enumerate() do uzyskania indeksu
            if (i+1) in excluded_obj:  # Pomijamy, jeśli indeks znajduje się w excluded_obj
                continue
            class_id, x_min, y_min, x_max, y_max = box
            # fill places where bbox exist with value 1 
            mask[y_min:y_max, x_min:x_max] = 1
        
    return mask

def calculate_pixel_precision(yolo_mask, gt_mask):
    # True Positives
    TP = np.sum(np.logical_and(yolo_mask == 1, gt_mask == 1))
    
    # False Positives
    FP = np.sum(np.logical_and(yolo_mask == 1, gt_mask == 0))
    
    # Calculate precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision



# Load dataset
dataset = load_dataset("rfeiglew/ThermalRefl_val", split="train")

#filter the dataset to get only question about reflection
filtered_dataset = dataset.filter(lambda item: item["question_id"] == 4)
print(filtered_dataset)

precision = 0

for element in filtered_dataset:
    image = element["yolo_detection_image"]
    im_width, im_height = image.size
    example_bbox = element["yolo_bbox"]
    gt_bbox = element["ground_truth_bbox"]

    # use vlm to exclude reflections
    #answer = generate_qwen_vlm_answer(model,tokenizer,image,element["question"])
    answer = generate_gemini_vlm_answer(image,element["question"])
    excluded_obj = ast.literal_eval(answer)
    #excluded_obj = ast.literal_eval(element["answer"])

    abs_yolo_bboxes = convert_yolo_bboxes_to_abs_bboxes(example_bbox,im_width,im_height)
    yolo_mask = create_mask(abs_yolo_bboxes, im_width,im_height,excluded_obj)


    gt_bboxes = convert_yolo_bboxes_to_abs_bboxes(gt_bbox,im_width,im_height)
    gt_mask =   create_mask(gt_bboxes, im_width,im_height)

    precision_one_elem = calculate_pixel_precision(yolo_mask, gt_mask)
    print(f"Precision of one element: {precision_one_elem}")
    print("Answer: ", answer)
    precision += precision_one_elem
    time.sleep(3)
    

precision = precision / filtered_dataset.num_rows
print(filtered_dataset.num_rows)

print(f"Precision of whole dataset: {precision}")

# im_mask = create_empty_mask(im_width,im_height)

# print(im_mask)



# print(dataset[0]["yolo_bbox"])

# example_bbox = dataset[0]["yolo_bbox"]

# print(example_bbox[0])
# print(example_bbox[0][1])

# value = example_bbox[0][1] + example_bbox[0][2]

# print(value)

