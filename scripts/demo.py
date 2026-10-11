#!/usr/bin/env python
"""Demo script for the QA Multi-Hop pipeline."""
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.graph import run_pipeline
from orchestrator.state import PipelineConfig


def run_demo():
    """Run demo with 3 examples."""
    config = PipelineConfig()
    
    examples = [
        {
            "name": "Multi-hop success",
            "question": "Which city is the capital of the country where the Eiffel Tower is located?",
            "expected": "Paris"
        },
        {
            "name": "Abstention",
            "question": "What is the exact temperature on Mars right now?",
            "expected": "Information insuffisante"
        },
        {
            "name": "Error handling - empty",
            "question": "",
            "expected": "Error"
        },
    ]
    
    print("=" * 60)
    print("QA MULTI-HOP DEMO")
    print("=" * 60)
    
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i}: {ex['name']} ---")
        print(f"Question: {ex['question']}")
        
        if not ex['question']:
            print("SKIPPED (empty question)")
            continue
        
        start = time.time()
        try:
            result = run_pipeline(ex['question'], config)
            latency = (time.time() - start) * 1000
            
            print(f"Answer: {result.answer}")
            print(f"Abstained: {result.abstained}")
            if result.abstained:
                print(f"Reason: {result.abstention_reason}")
            print(f"Sources cited: {len(result.sources)}")
            for j, src in enumerate(result.sources[:3], 1):
                print(f"  [{j}] ({src.h}, {src.r}, {src.t}) score={src.score:.3f}")
            print(f"Latency: {latency:.0f}ms")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)


def run_curl_examples():
    """Print curl commands for manual testing."""
    print("\nManual testing with curl:")
    print("-" * 40)
    
    commands = [
        ('Multi-hop', 'What is the capital of France?'),
        ('Abstention', 'What is the meaning of life?'),
        ('HotpotQA style', 'When did the park at which Tivolis Koncertsal is located open?'),
    ]
    
    for name, question in commands:
        escaped = question.replace('"', '\\"')
        print(f'# {name}')
        print(f'curl -X POST "http://localhost:8000/ask" \\')
        print(f'  -H "Content-Type: application/json" \\')
        print(f'  -d \'{{"question": "{escaped}"}}\'')
        print()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Run programmatic demo
    run_demo()
    
    # Show curl examples
    run_curl_examples()