"""Construction de prompts pour LLM à partir de triplets KG."""

from typing import List, Dict, Any


def build_prompt_from_triplets(question: str, triplets: List[Dict[str, Any]], max_triplets: int = 150) -> str:
    """Construit un prompt à partir de triplets KG.

    Trie par label décroissant (1 d'abord) puis ordre d'apparition.
    Numérote [1..n], inclut la phrase source.
    Tronque à max_triplets triplets max.

    Args:
        question: La question utilisateur
        triplets: Liste de dicts avec clés h, r, t, source_title, sentence, label
        max_triplets: Nombre max de triplets à inclure

    Returns:
        Prompt formaté
    """
    # Trier: label=1 d'abord, puis ordre original (index)
    sorted_triplets = sorted(
        enumerate(triplets),
        key=lambda x: (-x[1].get("label", 0), x[0])
    )

    # Tronquer
    sorted_triplets = sorted_triplets[:max_triplets]

    lines = [f"Question: {question}\n", "Contexte (triplets de connaissance):"]

    for idx, (_, t) in enumerate(sorted_triplets, 1):
        h = t.get("h", "")
        r = t.get("r", "")
        t_obj = t.get("t", "")
        sentence = t.get("sentence", "")
        label = t.get("label", 0)

        label_str = " ✓" if label == 1 else ""
        line = f"[{idx}] ({h}, {r}, {t_obj}){label_str}"
        if sentence:
            line += f" — Source: \"{sentence[:200]}\""
        lines.append(line)

    lines.append("\nRéponse:")
    return "\n".join(lines)


if __name__ == "__main__":
    # Test avec 3 triplets factices (1 label=1)
    dummy_triplets = [
        {
            "h": "tivoli gardens",
            "r": "amusement park",
            "t": "instance of",
            "source_title": "Tivoli Gardens",
            "sentence": "Tivoli Gardens is a famous amusement park in Copenhagen.",
            "label": 1,
        },
        {
            "h": "tivoli gardens",
            "r": "opened",
            "t": "15 august 1843",
            "source_title": "Tivoli Gardens",
            "sentence": "The park opened on 15 August 1843.",
            "label": 0,
        },
        {
            "h": "copenhagen",
            "r": "located in",
            "t": "denmark",
            "source_title": "Tivoli Gardens",
            "sentence": "Tivoli Gardens is located in Copenhagen, Denmark.",
            "label": 0,
        },
    ]

    question = "When did Tivoli Gardens open?"
    prompt = build_prompt_from_triplets(question, dummy_triplets)
    print("=== PROMPT GÉNÉRÉ ===")
    print(prompt)