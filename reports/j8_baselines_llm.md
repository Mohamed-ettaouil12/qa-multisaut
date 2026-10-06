# Baselines LLM sur Val - J8

Modèle: Qwen/Qwen2.5-1.5B-Instruct (4-bit)
Max new tokens: 64
Seed: 42

| Config           | EM     | F1     | Format OK % | Insuffisant à tort % |
| ---------------- | ------ | ------ | ----------- | -------------------- |
| C1_sans_contexte | 0.0000 | 0.0183 | 0.0         | 0.0                  |
| C2_top5_cosinus  | 0.0000 | 0.0232 | 0.0         | 0.0                  |
| C3_gnn_top5      | 0.0000 | 0.0242 | 0.0         | 0.0                  |
