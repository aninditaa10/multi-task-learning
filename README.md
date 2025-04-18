
# Sentence Transformers and Multi-Task Learning

This repository contains a lightweight, Dockerized implementation of a multi-task learning (MTL) model built on top of the `all-MiniLM-L6-v2` SentenceTransformer. It is designed to handle two NLP tasks simultaneously:

- **Task A:** Sentence / Topic Classification (e.g., Topic categorization)
- **Task B:** Sentiment Analysis (Positive / Neutral / Negative)

This code is fully containerized using Docker and tested in GitHub Codespaces.

---

##  Project Highlights

- 🔗 **Shared Transformer Encoder**: Utilizes `all-MiniLM-L6-v2` for generating sentence embeddings.
- 🔀 **Task-Specific Heads**: Two independent classification heads for Task A and Task B.
- 🧠 **Transfer Learning Ready**: Freezes base layers (0 to 3), fine-tunes task heads, final and final-1 layers (i.e. 4 and 5).
- 🐳 **Docker Compatible**: Easily deployable and reproducible in containerized environments.

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
# Build Docker image (note: tag must be lowercase)
docker build -t mlt_task .

# Run container
docker run --rm mlt_task
```
