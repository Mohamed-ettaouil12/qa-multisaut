"""Entraînement QLoRA SFT Run 2 (J10) sur dataset sft_train.jsonl.

- Modèle : Qwen/Qwen2.5-1.5B-Instruct (4-bit nf4)
- LoRA : r configurable (défaut 32), alpha=2*r, dropout=0.05
- Optimiseur : paged_adamw_8bit lr configurable (défaut 2e-4)
- Scheduler : cosine avec warmup 3%
- Batch size effectif 8 (per_device 2 x accumulation 4)
- Epochs : configurable (défaut 4)
- Max seq len : 512
- Gradient checkpointing
- Completion-only loss
- Checkpoint toutes les 100 steps dans checkpoints/qlora_run2/
- Logs MLflow

Usage:
    python src/llm/train_qlora.py --lora_r 32 --lr 2e-4 --epochs 4
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (default 2*r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qlora_run2", help="Output directory")
    parser.add_argument("--run_name", type=str, default="qlora_run2", help="MLflow run name")
    return parser.parse_args()


SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


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


def try_train(model_name: str, max_seq_len: int, batch_size: int, grad_accum: int,
              lora_r: int, lora_alpha: int, lora_dropout: float, lr: float, epochs: int,
              warmup_ratio: float, output_dir: str, run_name: str):
    """Tente l'entraînement, retourne (success, trainer, model, tokenizer)."""
    print(f"\n=== Tentative: {model_name}, max_seq_len={max_seq_len}, batch={batch_size}, accum={grad_accum} ===")
    print(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, lr={lr}, epochs={epochs}")

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
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_sft_dataset("data/processed/sft_train.jsonl")
    print(f"Dataset: {len(train_dataset)} exemples")

    total_steps = (len(train_dataset) // (batch_size * grad_accum)) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
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
        run_name=run_name,
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
        callbacks=[MLflowCallback(), CheckpointCallback(output_dir, save_steps=100)],
    )

    try:
        trainer.train()
        return True, trainer, model, tokenizer
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM: {e}")
        torch.cuda.empty_cache()
        return False, None, None, None


def main():
    args = parse_args()
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r

    set_seed()
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM totale: {total_mem:.1f} GB")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("qa-multisaut")
    mlflow.start_run(run_name=args.run_name)

    mlflow.log_params({
        "model": MODEL_NAME,
        "max_seq_length": args.max_seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "lora_r": args.lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": ",".join(TARGET_MODULES),
        "seed": SEED,
        "optim": "paged_adamw_8bit",
        "completion_only_loss": True,
    })

    # Tentative 1: Qwen2.5-1.5B, seq_len 512, batch 2 accum 4 (eff 8)
    success, trainer, model, tokenizer = try_train(
        MODEL_NAME, args.max_seq_len, args.batch_size, args.grad_accum,
        args.lora_r, lora_alpha, args.lora_dropout, args.lr, args.epochs,
        args.warmup_ratio, args.output_dir, args.run_name
    )

    # Tentative 2: Qwen2.5-1.5B, seq_len 384, batch 1 accum 16 (eff 16)
    if not success:
        print("\n--- Fallback: max_seq_len=384, batch=1, accum=16 ---")
        success, trainer, model, tokenizer = try_train(
            MODEL_NAME, 384, 1, 16,
            args.lora_r, lora_alpha, args.lora_dropout, args.lr, args.epochs,
            args.warmup_ratio, args.output_dir, args.run_name
        )
        if success:
            mlflow.log_param("fallback", "max_seq_len=384, batch=1, accum=16")

    # Tentative 3: Qwen2.5-0.5B
    if not success:
        print("\n--- Fallback: Qwen2.5-0.5B-Instruct ---")
        success, trainer, model, tokenizer = try_train(
            FALLBACK_MODEL, 384, 1, 16,
            args.lora_r, lora_alpha, args.lora_dropout, args.lr, args.epochs,
            args.warmup_ratio, args.output_dir, args.run_name
        )
        if success:
            mlflow.log_param("fallback", "model=Qwen2.5-0.5B-Instruct")

    if not success:
        print("ÉCHEC: Impossible d'entraîner même avec les fallbacks")
        mlflow.end_run()
        return

    print("\n=== Sauvegarde adaptateurs LoRA finaux ===")
    final_dir = Path(args.output_dir) / "final"
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
    print("\nEntraînement QLoRA Run 2 terminé !")


if __name__ == "__main__":
    main()