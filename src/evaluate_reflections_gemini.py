# Import libraries
import google.generativeai as genai
import re
from PIL import Image
import cv2
import numpy as np
from datasets import load_dataset
from unsloth import FastVisionModel
from transformers import TextStreamer
import torch
import time
import pandas as pd


API_KEY = "<ADD your API key here>"
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(model_name='gemini-1.5-flash')



# Load dataset
dataset = load_dataset("rfeiglew/ThermalRefl_val", split="train")


# Iterate through dataset and compare answers
results = []

for example in dataset:
    start_time = time.time()

    image = example["yolo_detection_image"]#.resize((640, 640))
    instruction = example["question"]
    expected_answer = example["answer"]


    generated_answer = model.generate_content([
        image,
        (
            instruction
        ),
    ])

    generated_answer = generated_answer.text

    end_time = time.time()
    iteration_time = end_time - start_time

    results.append({
        "question": instruction,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "time_taken" : iteration_time
    })

    print("Question:", instruction)
    print("Expected Answer:", expected_answer)
    print("Generated Answer:", generated_answer)
    print("Time Taken (s):", iteration_time)
    print("-" * 50)

# Save results to Excel
df = pd.DataFrame(results)
df.to_excel("model_results_gemini.xlsx", index=False)

# Print results
for result in results:
    print("Question:", result["question"])
    print("Expected Answer:", result["expected_answer"])
    print("Generated Answer:", result["generated_answer"])
    print("Time Taken (s):", result["time_taken"])
    print("-" * 50)
