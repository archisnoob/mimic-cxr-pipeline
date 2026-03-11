# MIMIC-CXR Data Pipeline
### Preprocessing & DataLoader Framework for Radiology Report Generation
> **Team Technicali** — Brain Dead, REVELATION 5.0 | ASCE, IIEST Shibpur | Feb 2026

---

## Overview

This repository contains a production-ready data ingestion and preprocessing pipeline for the **MIMIC-CXR** dataset, designed to prepare chest X-ray images and paired radiology reports for training vision-language models for automated report generation.

The pipeline handles the full preprocessing lifecycle — from raw CSV ingestion to batched, GPU-ready tensors — built entirely with PyTorch and HuggingFace Transformers.

---

## Problem Context

Radiology report generation is a challenging vision-language task. Before any model can be trained, a robust data pipeline must handle:

- **Inconsistent file paths** — MIMIC-CXR stores image paths as relative strings inside stringified Python lists, requiring careful parsing and resolution
- **Noisy annotations** — the dataset contains both ground-truth findings (`findings_gt`) and augmented noisy variants (`findings_noisy`), which must be separated and scored
- **Multi-image records** — a single patient record can contain multiple X-ray images, requiring a custom collation strategy
- **Text quality filtering** — short or null report entries must be detected and dropped before training

This pipeline solves all of the above.

---

## Dataset

**MIMIC-CXR** (Medical Information Mart for Intensive Care — Chest X-Ray)

| Split | Records | Source |
|-------|---------|--------|
| Train | ~500 | `mimic_cxr_aug_train.csv` |
| Validation | ~64,000 | `mimic_cxr_aug_validate.csv` |

- Images: PA (Posteroanterior) and AP (Anteroposterior) chest X-ray views
- Text: Free-text radiology findings (`findings_gt`) and augmented noisy variants (`findings_noisy`)
- Downloaded via `kagglehub` from `simhadrisadaram/mimic-cxr-dataset`

---

## Pipeline Architecture

```
Raw CSV Files
     │
     ▼
1. Data Loading & Column Cleanup
     │  → Drop junk index columns (Unnamed: 0, Unnamed: 0.1)
     │  → Rename columns to consistent schema
     ▼
2. Image Path Resolution
     │  → Parse stringified Python lists from CSV cells
     │  → Strip leading 'files/' prefix from relative paths
     │  → Resolve to absolute paths under IMAGE_ROOT
     │  → Validate all paths exist via assert checks
     ▼
3. Text Cleaning
     │  → Strip whitespace from findings_gt and findings_noisy
     │  → Drop records with null or too-short reports (< 20 chars)
     ▼
4. Quality Scoring
     │  → Assign gt_score = 1.0 for ground-truth findings
     │  → Assign noisy_score = 0.0 for augmented noisy findings
     ▼
5. PyTorch Dataset (MIMICCXRDataset)
     │  → Load and transform images (Resize 224×224, Normalize)
     │  → Tokenize findings via Flan-T5 tokenizer (max 512 tokens)
     │  → Support switching between gt and noisy text via use_noisy flag
     ▼
6. Custom Collate Function (mimic_cxr_collate)
     │  → Pad input_ids and attention_mask sequences to batch max length
     │  → Stack multi-image tensors per sample
     │  → Stack quality scores per batch
     ▼
GPU-Ready Batched Tensors
```

---

## Key Components

### `get_image_transform()`
Standard ImageNet-style preprocessing applied to every chest X-ray:
- Resize to 224×224
- Convert to tensor
- Normalize with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

### `resolve_image_paths()`
Handles the core path resolution bug in the raw MIMIC-CXR CSV:
- Safely parses both raw string and stringified list formats
- Strips the `files/` prefix that is prepended in the raw dataset
- Resolves each path relative to `IMAGE_ROOT`

### `MIMICCXRDataset`
Custom PyTorch `Dataset` class supporting:
- Multi-image loading per patient record
- Configurable image size via `image_size` parameter
- Switchable text source (`findings_gt` vs `findings_noisy`) via `use_noisy` flag
- Flan-T5 tokenization with truncation at 512 tokens

### `mimic_cxr_collate()`
Custom collate function for the DataLoader:
- Handles variable-length token sequences via `pad_sequence`
- Preserves multi-image structure per sample
- Stacks quality score tensors for potential score-conditioned training

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data handling | pandas, numpy |
| Image processing | torchvision, Pillow |
| Deep learning framework | PyTorch |
| Tokenizer | HuggingFace Transformers (Flan-T5) |
| Progress tracking | tqdm |
| Dataset download | kagglehub |
| Environment | Google Colab (T4 GPU) |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/archisnoob/mimic-cxr-report-pipeline.git
cd mimic-cxr-report-pipeline
```

### 2. Install dependencies
```bash
pip install torch torchvision transformers pillow pandas numpy tqdm kagglehub einops
```

### 3. Run the notebook
Open `Chest_X_ray.ipynb` in Google Colab (T4 GPU recommended) and run all cells sequentially.

The pipeline will:
1. Download the MIMIC-CXR dataset automatically via kagglehub
2. Clean and preprocess all CSV records
3. Resolve and validate all image paths
4. Initialize the Dataset and DataLoader
5. Verify a sample batch loads correctly

---

## Repository Structure

```
mimic-cxr-report-pipeline/
│
├── Chest_X_ray.ipynb        # Main Colab notebook (full pipeline)
├── chest_x_ray_pipeline.py  # Equivalent .py script
└── README.md                # This file
```

---

## Design Decisions

**Why Flan-T5 as tokenizer?**
Flan-T5 is a strong seq2seq backbone well-suited for text generation tasks. The tokenizer is used here purely for preprocessing — tokenizing radiology findings into input IDs and attention masks ready for a T5-family decoder.

**Why support noisy findings?**
The MIMIC-CXR augmented dataset includes both clean ground-truth reports and noisy augmented variants. The `use_noisy` flag and quality scoring system (`gt_score`, `noisy_score`) are designed to support noise-robust or score-conditioned training objectives in future model training stages.

**Why a custom collate function?**
Standard PyTorch collation cannot handle variable-length image lists per record. The custom `mimic_cxr_collate` function explicitly handles multi-image batching while padding text sequences to uniform length.

---

## Team

- **Archisman Ghosh** — IIEST Shibpur, B.Tech Electrical Engineering
- **Ridhibrata Das**
- **Rishik Pal**

---

## Competition Context

This pipeline was developed for **Brain Dead 2026** at **REVELATION 5.0**, organized by the Academic Society of Computer Engineers (ASCE), Department of Computer Science & Technology, IIEST Shibpur (Feb 2026).

---

## License

This project is intended for academic and research purposes only. MIMIC-CXR data access requires credentialed registration on PhysioNet.
