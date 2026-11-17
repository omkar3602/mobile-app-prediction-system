# Mobile App Permission Prediction

Can you tell what an Android app will ask access to — camera, location, contacts —
just from its Play Store description? Fine-tuned transformer classifiers for the 30 most
common permissions across 5,400+ apps, benchmarked against zero- and few-shot LLMs.

## Highlights
- **Fine-tuned DeBERTa-v3-small (~184M params): 0.91 avg F1** across 30 permissions
- **~4× the F1 of prompted 7B LLMs** (LLaMA-2-7B 0.23, Mistral-7B 0.24) on the same binary task
- Error analysis: most mistakes are **false negatives** caused by vague descriptions,
  not model capacity

## Approach
- **Data:** 5,404 web-scraped Google Play apps with multi-label permission annotations
  (`data/`), filtered to English (`langdetect`), HTML stripped, stopwords removed.
  Labels = the 30 most frequent permissions (`data/sorted_permissions.csv`).
- **Fine-tuned classifiers:** one binary classifier per permission (80/20 split,
  3 epochs, lr 2e-4) — DistilBERT first, then `microsoft/deberta-v3-small`.
- **Class imbalance:** minority-class oversampling with WordNet synonym augmentation (`nlpaug`).
- **LLM baselines:** LLaMA-2-7B, Mistral-7B (4-bit, `bitsandbytes`) and FLAN-T5-Large,
  prompted zero-shot and few-shot, for both binary and multi-label prediction.

## Results

**Binary classification (one model per permission, averaged over 30 permissions)**

| Model | Params | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LLaMA-2-7B | 7B | 0.48 | 0.43 | 0.20 | 0.23 |
| Mistral-7B | 7B | 0.58 | 0.56 | 0.21 | 0.24 |
| FLAN-T5-Large | ~0.78B | 0.54 | 0.53 | 0.83 | 0.58 |
| DistilBERT | ~66M | 0.79 | 0.80 | 0.79 | 0.78 |
| **DeBERTa-v3-small** | **~184M** | **0.93** | **0.93** | **0.92** | **0.91** |

**Multi-label classification with LLMs (all 30 permissions in one prompt)**

| Model | Setting | Hamming loss ↓ | Jaccard | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| LLaMA-2-7B | zero-shot | 0.51 | 0.15 | 0.35 | 0.33 | 0.34 |
| Mistral-7B | zero-shot | 0.48 | 0.15 | 0.34 | 0.23 | 0.21 |
| LLaMA-2-7B | few-shot | 0.51 | 0.22 | 0.45 | 0.37 | 0.32 |
| Mistral-7B | few-shot | 0.45 | 0.12 | 0.45 | 0.15 | 0.19 |

## Findings
- **Small fine-tuned encoders win.** A 184M-parameter DeBERTa beats 7B LLMs by a wide
  margin; prompted LLMs have especially low recall (≈0.20) and miss most permissions.
- **Frequency drives performance.** Common permissions (full network access, storage,
  photos) score far higher than rare ones (e.g. *Phone: read call log*), which lack training data.
- **Conservative predictions.** Per-permission, the fine-tuned models lean high-precision /
  lower-recall — e.g. *full network access*: precision > 0.99, recall 0.63, F1 0.77.
- **Ambiguity is the main error source.** Manual inspection of 5 misclassified apps per
  permission: false negatives dominate for Bluetooth, Wi-Fi and storage, where descriptions
  only imply the feature ("background sync", "connectivity").

## Repo layout
- `binary_classification.ipynb` — preprocessing, per-permission fine-tuning
  (DistilBERT → DeBERTa), augmentation and error analysis
- `LLMs_binary_classification.ipynb` — LLM pipeline, one yes/no prompt per permission
- `LLMS_multilabel_classification.ipynb` — LLM pipeline, all 30 permissions in one prompt,
  zero- and few-shot
- `data/` — cleaned dataset, top-30 subset, permission frequency list
