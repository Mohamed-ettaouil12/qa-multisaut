"""Entraînement QLoRA SFT (J8) sur dataset sft_train.jsonl.

- Modèle : Qwen/Qwen2.5-1.5B-Instruct (4-bit nf4)
- LoRA : r=8, alpha=16, dropout=0.1, target modules q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
- Optimiseur : AdamW lr=2e-4, weight_decay=0.01
- Scheduler : cosine avec warmup 10%
- Batch size 1 + gradient accumulation 16 (effective batch 16)
- Epochs : 3
- Sauvegarde : checkpoints/llm_sft_lora/ (adapters)
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from pathlib import Path
from typing import Dict, List

import mlflow
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4
WEIGHT_DECAY = 0.01
EPOCHS = 3
WARMUP_RATIO = 0.1
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
OUTPUT_DIR = "checkpoints/llm_sft_lora"


def set_seed(seed: int = SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sft_dataset(path: str) -> Dataset:
    """Charge le dataset SFT au format prompt/completion."""
    data = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            data.append({
                "prompt": d["prompt"],
                "completion": d["completion"],
            })
    return Dataset.from_list(data)


def formatting_func(example: Dict) -> str:
    """Formate prompt + completion pour SFTTrainer."""
    return example["prompt"] + " " + example["completion"] + "<|im_end|>"


class MLflowCallback(TrainerCallback):
    """Callback pour logger les métriques dans MLflow."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=state.global_step)


def main():
    set_seed()
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"PyTorch: {torch.__version__}")

    # Vérifier VRAM
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM totale: {total_mem:.1f} GB")
        if total_mem < 3.5:
            print("ATTENTION: VRAM < 3.5 GB, risque OOM")

    # MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("qa-multisaut")
    mlflow.start_run(run_name="llm_sft_qlora")

    mlflow.log_params({
        "model": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": ",".join(TARGET_MODULES),
        "seed": SEED,
    })

    # Charger tokenizer
    print("Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Charger modèle 4-bit
    print("Chargement modèle 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Vérifier que bitsandbytes a bien chargé en 4-bit
    if not hasattr(model, "is_loaded_in_4bit") or not model.is_loaded_in_4bit:
        print("ERREUR: Modèle non chargé en 4-bit (bitsandbytes CUDA échec)")
        mlflow.end_run()
        return

    print(f"Modèle chargé en 4-bit: {model.is_loaded_in_4bit}")
    print(f"Device: {next(model.parameters()).device}")

    # Préparer pour k-bit training
    model = prepare_model_for_kbit_training(model)

    # Config LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    print("Chargement dataset SFT...")
    train_dataset = load_sft_dataset("data/processed/sft_train.jsonl")
    print(f"  {len(train_dataset)} exemples")

    # Training args (SFTConfig)
    total_steps = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="mlflow",
        run_name="llm_sft_qlora",
        seed=SEED,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        completion_only_loss=False,
        dataset_text_field="text",  # sera ignoré car formatting_func fourni
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        callbacks=[MLflowCallback()],
    )

    print("\n=== Début entraînement QLoRA ===")
    trainer.train()

    print("\n=== Sauvegarde adaptateurs LoRA ===")
    final_dir = Path(OUTPUT_DIR) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Adaptateurs sauvés: {final_dir}")

    # Test rapide
    print("\n=== Test génération ===")
    test_prompt = "Question: What is 2+2?\nRéponse:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    print(f"Prompt: {test_prompt}")
    print(f"Réponse: {tokenizer.decode(generated, skip_special_tokens=True).strip()}")

    mlflow.end_run()
    print("\nEntraînement terminé !")


if __name__ == "__main__":
    main()