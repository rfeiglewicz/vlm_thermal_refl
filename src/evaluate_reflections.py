from datasets import load_dataset
from unsloth import FastVisionModel
from transformers import TextStreamer
import torch
import time
import pandas as pd

# Load dataset
dataset = load_dataset("rfeiglew/ThermalRefl_val", split="train")

# Load model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    # model_name="rfeiglew/qwen2vl_refl_2b",
    model_name = "rfeiglew/qwen2_5vl_refl_3b",
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

# Initialize text streamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# Iterate through dataset and compare answers
results = []

for example in dataset:
    start_time = time.time()

    image = example["yolo_detection_image"]#.resize((640, 640))
    instruction = example["question"]
    expected_answer = example["answer"]

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=0.1, min_p=0.1)
    generated_answer = tokenizer.decode(outputs[0, len(inputs.input_ids[0]):], skip_special_tokens=True)

    end_time = time.time()
    iteration_time = end_time - start_time

    results.append({
        "question": instruction,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "time_taken" : iteration_time
    })

# Save results to Excel
df = pd.DataFrame(results)
df.to_excel("model_results.xlsx", index=False)

# Print results
for result in results:
    print("Question:", result["question"])
    print("Expected Answer:", result["expected_answer"])
    print("Generated Answer:", result["generated_answer"])
    print("Time Taken (s):", result["time_taken"])
    print("-" * 50)
