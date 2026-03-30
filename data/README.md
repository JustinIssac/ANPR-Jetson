# Data

## anpr_database.db — SQLite Database

Pre-seeded with 6 test plates and access log entries from the demo session.

**View contents:**
```bash
bash scripts/show_db.sh
```

**Registered plates (pre-seeded):**

| Plate | Status | Owner |
|-------|--------|-------|
| KL01CA2555 | AUTHORIZED | Test Vehicle 1 |
| MH12AB1234 | FLAGGED | Flagged Vehicle |
| PGMN112 | AUTHORIZED | Montenegro Car |
| PRENUP | AUTHORIZED | Camaro |
| DZ17YXR | AUTHORIZED | Blue Subaru |
| PU18BES | AUTHORIZED | Peugeot |

## Training Dataset

**Source:** [ANPR License Plate Detection — Kaggle (harshitsingh09)](https://www.kaggle.com/datasets/harshitsingh09/anpr-license-plate-detection)

- Format: YOLO (pre-annotated bounding boxes)
- Split: 80% train / 10% val / 10% test

Download and place at `data/YOLO_dataset/` before running `notebooks/phase1_training.ipynb`.
