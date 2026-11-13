# Dataset

## JM1 — NASA Metrics Data Program

The JM1 dataset cannot be included in this repository due to size.

### Download Instructions

**Option 1 — OpenML (easiest)**
1. Go to https://openml.org/d/1053
2. Click "Download" → CSV
3. Save as `jm1.csv` in this `data/` folder

**Option 2 — Direct link**
```
https://www.openml.org/data/get_csv/53939/jm1.arff
```

**Option 3 — UCI / Kaggle search**
Search "NASA JM1 defect prediction dataset"

### Dataset Info

| Property | Value |
|----------|-------|
| Rows | 13,204 |
| Columns | 22 (21 features + 1 target) |
| Target column | `defects` (True/False) |
| Class balance | 84.1% non-defective, 15.9% defective |

### Expected file path
Place the downloaded file at:
```
data/jm1.csv
```
