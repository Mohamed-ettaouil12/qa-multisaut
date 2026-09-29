"""Extraction de triplets avec REBEL (Babelscape/rebel-large)"""
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_extractor():
    """Charge REBEL-large en fp16 sur CUDA (fallback CPU)."""
    model_name = "Babelscape/rebel-large"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def parse_rebel_output(text: str) -> list[dict]:
    """Parse la sortie REBEL : '<triplet> sujet <subj> objet <obj> <triplet> ...'"""
    triplets = []
    pattern = r"<triplet>\s*([^<]+?)\s*<subj>\s*([^<]+?)\s*<obj>\s*([^<]+?)(?=\s*<triplet>|\s*$)"
    matches = re.findall(pattern, text)
    for match in matches:
        if len(match) == 3:
            h, r, t = match[0].strip(), match[1].strip(), match[2].strip()
            if h and r and t:
                triplets.append({"h": h, "r": r, "t": t})
    return triplets


def extract_paragraph(model, tokenizer, texts: list[str], batch_size: int = 16) -> list[list[dict]]:
    """Extrait les triplets pour une liste de textes (batch 16, beams=3, max_len=256)."""
    all_triplets = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=3,
                max_length=256,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        for text in decoded:
            triplets = parse_rebel_output(text)
            all_triplets.append(triplets)

    return all_triplets


if __name__ == "__main__":
    model, tokenizer = load_extractor()
    sample_text = "Paris est la capitale de la France. Emmanuel Macron est le président."
    triplets = extract_paragraph(model, tokenizer, [sample_text])
    print(f"Triplets extraits: {triplets}")