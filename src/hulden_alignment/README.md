# alignment
Module related to the Hulden alignment of a string-to-string dataset.

Usage:
```python
from src.data import create_data_files
from src.alignment import run_alignment, train_alignment_predictor

data_files = create_data_files(
    task="inflection",
    dataset="swe",
    has_features=True,
)

# Run alignment on some number of files in TSV format
alignment.run_alignment(data_files)

# Train transformer to predict alignment


```
