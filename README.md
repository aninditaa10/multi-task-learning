
# Sentence Transformers and Multi-Task Learning

This repository contains an implementation of a multi-task learning (MTL) model built on top of the `all-MiniLM-L6-v2` SentenceTransformer. It is designed to handle two NLP tasks simultaneously:

- **Task A:** Sentence / Topic Classification (e.g., Topic categorization)
- **Task B:** Sentiment Analysis (Positive / Neutral / Negative)

This code is containerized using Docker and tested in GitHub Codespaces.

---

##  Project Highlights

- 🔗 **Shared Transformer Encoder**: Utilizes `all-MiniLM-L6-v2` for generating sentence embeddings.
- 🔀 **Task-Specific Heads**: Two independent classification heads for Task A and Task B.
- 🧠 **Transfer Learning Ready**: Freezes base layers (0 to 3), fine-tunes task heads, final and final-1 layers (i.e. 4 and 5).
---

## Tasks Implemented

| Task   | Description                                        |
| ------ | -------------------------------------------------- |
| Task 1 | Sentence encoding with SentenceTransformer         |
| Task 2 | Multi-task expansion: Shared encoder, dual heads   |
| Task 3 | Transfer learning strategy with selective freezing |
| Task 4 | Complete training loop implementation              |

---

##  Installation & Execution

### ➤ In GitHub Codespaces or any Docker-enabled terminal:

```bash
# Build Docker image with the file path
docker build -t multitask_1 /workspaces/multi-task-learning 

# Run 
docker run --rm multitask_1
```
### Output obtained: 
```
Sentence 1: Machine learning is transforming the world.
Embedding (first 5 values): [ 0.10397332 -0.23789866  0.60727847 -0.07666507  0.4981051 ]
Total length: 384

Sentence 2: I love exploring deep learning models.
Embedding (first 5 values): [-0.26113355 -0.94204414  0.40685245  0.0758177   0.30714193]
Total length: 384

Sentence 3: I enjoy going for Marathon but this weather does not allow me to do so.
Embedding (first 5 values): [0.13154854 0.01713244 0.5616088  0.38909754 0.05436677]
Total length: 384


=== TRAIN ===
The movie was fantastic and thrilling.
→ Topic: Entertainment | Sentiment: Positive

The market is crashing rapidly.
→ Topic: Finance | Sentiment: Negative

Python is a great programming language.
→ Topic: Technology | Sentiment: Positive


=== UNSEEN ===
Oil prices across the world are hurting everyone's wallet.
→ Topic: Finance | Sentiment: Negative

Mission Impossible latest movie was superb!
→ Topic: Entertainment | Sentiment: Positive

I love building AI solution and agents which addresses key society challenges.
→ Topic: Technology | Sentiment: Positive
```
