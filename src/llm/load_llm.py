"""Chargement LLM pour inférence et fine-tuning."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_base_model(device: str = "cuda", use_4bit: bool = True):
    """Charge le modèle de base (4-bit QLoRA-ready)."""
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def load_qlora_model(adapter_path: str, device: str = "cuda"):
    """Charge un modèle avec adaptateurs LoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = PeftModel.from_pretrained(base_model, adapter_path, device_map={"": device})
    return model, tokenizer


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    """Génère une réponse (greedy)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Décoder seulement la partie générée
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


if __name__ == "__main__":
    # Test rapide
    model, tokenizer = load_base_model("cuda", use_4bit=True)
    print(f"Modèle chargé: {MODEL_NAME}")
    print(f"Device: {model.device}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    test_prompt = "Question: What is 2+2?\nRéponse:"
    print(f"\nTest: {test_prompt}")
    ans = generate_answer(model, tokenizer, test_prompt)
    print(f"Réponse: {ans}")