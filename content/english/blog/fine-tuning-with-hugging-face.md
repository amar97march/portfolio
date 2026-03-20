---
title: "Tool Spotlight: Fine-Tuning Models with Hugging Face Transformers"
date: 2026-11-26T10:00:00+05:30
draft: false
description: "A hands-on guide to fine-tuning LLMs using the Hugging Face ecosystem — Transformers, PEFT, TRL, and Datasets — with complete code examples."
tags: ["Hugging Face", "Fine-Tuning", "Transformers", "PEFT", "Python", "Tutorial"]
categories: ["Fine-Tuning"]
image: "/images/blogs/pool-llm/1.jpg"
keywords: ["Hugging Face fine-tuning", "transformers library", "TRL trainer", "PEFT tutorial", "fine-tune Llama", "Hugging Face tutorial"]
---

If there is one ecosystem that has done more to democratize AI than any other, it is **Hugging Face**. Their suite of libraries — Transformers, PEFT, TRL, Datasets, and Accelerate — provides everything you need to fine-tune LLMs, from data preparation to model deployment.

In this post, we will walk through a complete fine-tuning workflow using the Hugging Face stack. By the end, you will have a working fine-tuned model ready for inference.

### The Hugging Face Stack

```
┌─────────────────────────────────────┐
│ Datasets    - Load and preprocess   │
│ Transformers - Models and tokenizers │
│ PEFT        - LoRA and adapters     │
│ TRL         - Training (SFT, DPO)   │
│ Accelerate  - Multi-GPU / mixed precision │
│ Hub         - Share and deploy      │
└─────────────────────────────────────┘
```

Each library handles a specific concern, and they integrate seamlessly.

### Setup

```bash
pip install torch transformers peft trl datasets accelerate bitsandbytes
```

You will also need a Hugging Face account and access token for gated models like Llama:

```bash
huggingface-cli login
```

### Step 1: Prepare Your Dataset

The first step is converting your data into the format the training pipeline expects.

```python
# prepare_data.py
from datasets import Dataset
import json

def load_training_data(file_path: str) -> Dataset:
    """Load JSONL training data into a Hugging Face Dataset."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    return Dataset.from_list(examples)

def format_for_sft(example):
    """Format a conversation example for SFT training."""
    messages = example['messages']
    formatted = ""
    for msg in messages:
        if msg['role'] == 'system':
            formatted += f"<|system|>\n{msg['content']}\n"
        elif msg['role'] == 'user':
            formatted += f"<|user|>\n{msg['content']}\n"
        elif msg['role'] == 'assistant':
            formatted += f"<|assistant|>\n{msg['content']}\n"
    return {"text": formatted}

# Load and format data
dataset = load_training_data("training_data.jsonl")
dataset = dataset.map(format_for_sft)

# Split into train and eval
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"Training examples: {len(train_dataset)}")
print(f"Evaluation examples: {len(eval_dataset)}")
```

**Using existing datasets from the Hub:**

```python
from datasets import load_dataset

# Load a popular fine-tuning dataset
dataset = load_dataset("tatsu-lab/alpaca")

# Or use a specific subset
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train[:5000]")
```

### Step 2: Load the Model with Quantization

```python
# model_setup.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def load_model_and_tokenizer():
    """Load model in 4-bit with tokenizer."""

    # Quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Faster attention
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
print(f"Model loaded: {MODEL_NAME}")
print(f"Model size: {model.get_memory_footprint() / 1e9:.1f} GB")
```

![Loading and quantizing large language models for training](/images/blogs/pool-llm/6.jpg)

### Step 3: Configure LoRA

```python
# lora_config.py
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def apply_lora(model):
    """Apply LoRA configuration to the model."""

    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,                          # Rank
        lora_alpha=32,                 # Scaling factor
        target_modules=[
            "q_proj", "v_proj",        # Attention
            "k_proj", "o_proj",        # Attention
            "gate_proj", "up_proj",    # MLP
            "down_proj",               # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

model = apply_lora(model)
```

### Step 4: Train with SFTTrainer

```python
# train.py
from transformers import TrainingArguments
from trl import SFTTrainer

def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Fine-tune the model using SFTTrainer."""

    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",           # Or "wandb" for experiment tracking
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=2048,
        dataset_text_field="text",
        packing=True,               # Pack short examples together
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the final adapter
    trainer.save_model("./final_adapter")
    print("Training complete! Adapter saved to ./final_adapter")

    return trainer

trainer = train_model(model, tokenizer, train_dataset, eval_dataset)
```

![Training with LoRA adapters and SFT optimization](/images/blogs/pool-llm/7.jpg)

### Step 5: Evaluate

```python
# evaluate.py
from peft import PeftModel
from transformers import pipeline

def evaluate_model(adapter_path: str, test_prompts: list):
    """Evaluate the fine-tuned model on test prompts."""

    # Load base model and apply adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
    )

    # Run test prompts
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        result = pipe(prompt)
        print(f"Response: {result[0]['generated_text']}")

test_prompts = [
    "How do I troubleshoot a 502 error on CloudBase?",
    "What are the steps to set up CI/CD for my application?",
    "How do I configure custom domains?",
]

evaluate_model("./final_adapter", test_prompts)
```

### Step 6: Merge and Deploy

```python
# deploy.py
from peft import PeftModel

def merge_and_save(adapter_path: str, output_path: str):
    """Merge LoRA adapter with base model for deployment."""

    # Load base model at full precision
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load and merge adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    # Save merged model
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")

merge_and_save("./final_adapter", "./merged_model")
```

**Push to Hugging Face Hub:**

```python
# Upload to Hub for easy sharing and deployment
model.push_to_hub("your-username/my-fine-tuned-model")
tokenizer.push_to_hub("your-username/my-fine-tuned-model")
```

### The Complete Script

```python
#!/usr/bin/env python3
"""Complete fine-tuning pipeline with Hugging Face."""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./fine_tuned_output"

# 1. Load and prepare data
dataset = load_dataset("json", data_files="training_data.jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

# 2. Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 3. Apply LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj",
                     "gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
))

# 4. Train
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset["train"], eval_dataset=dataset["test"],
    max_seq_length=2048, dataset_text_field="text",
    args=TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, logging_steps=10,
        save_strategy="epoch", eval_strategy="epoch",
        load_best_model_at_end=True,
    ),
)
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
print("Done! Fine-tuned adapter saved.")
```

![Merging adapters and deploying fine-tuned models](/images/blogs/pool-llm/8.jpg)

### Tips From Production

1. **Use Weights & Biases (wandb) for tracking**: Set `report_to="wandb"` in TrainingArguments. Track loss curves, learning rates, and evaluation metrics across experiments.

2. **Enable gradient checkpointing**: Always use `gradient_checkpointing=True` when memory is limited. It trades 20% slower training for 50%+ memory savings.

3. **Pack short examples**: The `packing=True` option in SFTTrainer concatenates short examples into a single sequence, dramatically improving GPU utilization.

4. **Version your adapters**: Use git or the Hugging Face Hub to version control your adapters. Each training run should produce a clearly labeled artifact.

5. **Start small, scale up**: Begin with a small dataset (100-500 examples) on a small model (7-8B). Validate your pipeline works end-to-end before scaling to larger datasets and models.

The Hugging Face ecosystem makes fine-tuning approachable for any developer. The tools handle the complexity of distributed training, mixed precision, and gradient management, letting you focus on what matters: your data and your use case.
