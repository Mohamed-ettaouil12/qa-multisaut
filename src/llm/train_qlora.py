"""Entraînement QLoRA SFT Run 1 (J8-J9) sur dataset sft_train.jsonl.

- Modèle : Qwen/Qwen2.5-1.5B-Instruct (4-bit nf4)
- LoRA : r=16, alpha=32, dropout=0.05, target modules q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Optimiseur : paged_adamw_8bit lr=2e-4
- Scheduler : cosine avec warmup 3%
- Batch size effectif 8 (per_device 1-2 x accumulation)
- Epochs : 2
- Max seq len : 512 (fallback 384 si OOM)
- Gradient checkpointing
- Completion-only loss
- Checkpoint toutes les 100 steps dans checkpoints/qlora_run1/
- Logs MLflow
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
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 512
FALLBACK_MAX_SEQ_LENGTH = 384
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 2e-4
EPOCHS = 2
WARMUP_RATIO = 0.03
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
OUTPUT_DIR = "checkpoints/qlora_run1"


def set_seed(seed: int = SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sft_dataset(path: str) -> Dataset:
    """Charge le dataset SFT au format prompt/completion et pré-formate pour completion_only_loss."""
    data = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # Pré-formatter: prompt + completion pour completion_only_loss
            text = d["prompt"] + " " + d["completion"] + " "
            data.append({"text": text})
    return Dataset.from_list(data)


class MLflowCallback(TrainerCallback):
    """Callback pour logger les métriques dans MLflow."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=state.global_step)


class CheckpointCallback(TrainerCallback):
    """Callback pour sauvegarder checkpoint toutes les 100 steps."""
    def __init__(self, output_dir: str, save_steps: int = 100):
        self.output_dir = Path(output_dir)
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            checkpoint_dir = self.output_dir / f"checkpoint-{state.global_step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            kwargs["model"].save_pretrained(str(checkpoint_dir))
            kwargs["processing_class"].save_pretrained(str(checkpoint_dir))
            print(f"Checkpoint sauvé: {checkpoint_dir}")


def try_train(model_name: str, max_seq_len: int, batch_size: int, grad_accum: int):
    """Tente l'entraînement, retourne (success, trainer, model, tokenizer)."""
    print(f"\n=== Tentative: {model_name}, max_seq_len={max_seq_len}, batch={batch_size}, accum={grad_accum} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if not hasattr(model, "is_loaded_in_4bit") or not model.is_loaded_in_4bit:
        print("ERREUR: Modèle non chargé en 4-bit")
        return False, None, None, None

    print(f"Modèle chargé en 4-bit: {model.is_loaded_in_4bit}")
    print(f"Device: {next(model.parameters()).device}")

    model = prepare_model_for_kbit_training(model)

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

    train_dataset = load_sft_dataset("data/processed/sft_train.jsonl")
    print(f"Dataset: {len(train_dataset)} exemples")

    total_steps = (len(train_dataset) // (batch_size * grad_accum)) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="mlflow",
        run_name="qlora_run1",
        seed=SEED,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        max_length=max_seq_len,
        packing=False,
        completion_only_loss=True,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[MLflowCallback(), CheckpointCallback(OUTPUT_DIR, save_steps=100)],
    )

    try:
        trainer.train()
        return True, trainer, model, tokenizer
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM: {e}")
        torch.cuda.empty_cache()
        return False, None, None, None


def main():
    set_seed()
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM totale: {total_mem:.1f} GB")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("qa-multisaut")
    mlflow.start_run(run_name="qlora_run1")

    mlflow.log_params({
        "model": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "lr": LR,
        "epochs": EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": ",".join(TARGET_MODULES),
        "seed": SEED,
        "optim": "paged_adamw_8bit",
        "completion_only_loss": True,
    })

    # Tentative 1: Qwen2.5-1.5B, seq_len 512, batch 2 accum 4 (eff 8)
    success, trainer, model, tokenizer = try_train(MODEL_NAME, MAX_SEQ_LENGTH, BATCH_SIZE, GRAD_ACCUM)

    # Tentative 2: Qwen2.5-1.5B, seq_len 384, batch 1 accum 16 (eff 16)
    if not success:
        print("\n--- Fallback: max_seq_len=384, batch=1, accum=16 ---")
        success, trainer, model, tokenizer = try_train(MODEL_NAME, FALLBACK_MAX_SEQ_LENGTH, 1, 16)
        if success:
            mlflow.log_param("fallback", "max_seq_len=384, batch=1, accum=16")

    # Tentative 3: Qwen2.5-0.5B
    if not success:
        print("\n--- Fallback: Qwen2.5-0.5B-Instruct ---")
        success, trainer, model, tokenizer = try_train(FALLBACK_MODEL, FALLBACK_MAX_SEQ_LENGTH, 1, 16)
        if success:
            mlflow.log_param("fallback", "model=Qwen2.5-0.5B-Instruct")

    if not success:
        print("ÉCHEC: Impossible d'entraîner même avec les fallbacks")
        mlflow.end_run()
        return

    print("\n=== Sauvegarde adaptateurs LoRA finaux ===")
    final_dir = Path(OUTPUT_DIR) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Adaptateurs sauvés: {final_dir}")

    # Test rapide : 10 questions val avec contexte GNN top-k
    print("\n=== Test rapide sur 10 questions val (contexte GNN top-k) ===")
    from src.llm.prompt import build_prompt_from_triplets
    from src.llm.load_llm import generate_answer
    import json
    from collections import defaultdict

    val_data = []
    with open("data/processed/val.jsonl") as f:
        for line in f:
            val_data.append(json.loads(line))

    qid_to_triplets = defaultdict(list)
    with open("data/processed/kg_triplets.jsonl") as f:
        for line in f:
            t = json.loads(line)
            qid_to_triplets[t["qid"]].append(t)

    gnn_topk = {}
    with open("data/processed/gnn_topk_val.jsonl") as f:
        for line in f:
            d = json.loads(line)
            gnn_topk[d["qid"]] = d["triplets"]

    for i, ex in enumerate(val_data[:10]):
        qid = ex["id"]
        question = ex["question"]
        gold = ex["answer"]

        triplets = qid_to_triplets.get(qid, [])
        gnn_triplets = gnn_topk.get(qid, [])

        prompt = build_prompt_from_triplets(question, gnn_triplets)
        response = generate_answer(model, tokenizer, prompt, max_new_tokens=64)

        print(f"\n--- Test {i+1} (qid: {qid}) ---")
        print(f"Question: {question}")
        print(f"Gold: {gold}")
        print(f"Réponse brute: {response}")

    mlflow.end_run()
    print("\nEntraînement QLoRA Run 1 terminé !")


if __name__ == "__main__":
    main()