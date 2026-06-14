#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import subprocess
import json
import os

model_id = "HuggingFaceTB/SmolLM2-135M"
local_dir = "models/SmolLM2-135M"

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(repo_id=model_id, local_dir=local_dir)

print(f"Loading {model_id} via transformers...")
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float32)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
print(f"Token IDs: {inputs.input_ids.tolist()[0]}")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].to(torch.float32).numpy()

top_k = 5
top_indices = logits.argsort()[-top_k:][::-1]
print("\nPyTorch Top 5 logits:")
for idx in top_indices:
    print(f"Token {idx}: {logits[idx]:.4f}")

print("\nPacking via rllm-cli...")
subprocess.run(["cargo", "run", "--release", "--bin", "rllm", "--", "pack", "models/SmolLM2-135M/model.safetensors", "--out", "models/SmolLM2-135M.rllm"], check=True)

print("\nRunning Rust engine...")
prompt_str = ",".join(map(str, inputs.input_ids.tolist()[0]))
rust_output = subprocess.run(["cargo", "run", "--release", "--bin", "llama-test", "--", "--model", "models/SmolLM2-135M.rllm", "--prompt", prompt_str], capture_output=True, text=True)

print("\nRust Output:")
print(rust_output.stdout)
if rust_output.stderr:
    print("Stderr:", rust_output.stderr)

