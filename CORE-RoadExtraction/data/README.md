# Data Directory

This directory is intentionally mostly empty in the repository. Satellite image datasets are too large to store in git.

## How to set up your data

### Option 1 — Custom dataset (flat layout)

Place your images and masks here:

```
data/raw/
├── images/
│   ├── tile_001.png
│   ├── tile_002.png
│   └── ...
└── masks/
    ├── tile_001.png   ← same filename as image
    ├── tile_002.png
    └── ...
```

Then run:
```bash
python scripts/prepare_data.py \
    --raw_dir data/raw \
    --out_dir data/processed \
    --format flat
```

### Option 2 — DeepGlobe dataset

Download from: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

Place the extracted folder under `data/raw/`, then:
```bash
python scripts/prepare_data.py \
    --raw_dir data/raw \
    --out_dir data/processed \
    --format deepglobe
```

### After preparation

`data/processed/` will contain:
```
data/processed/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Update `configs/default.yaml` with these paths before training.
