# AGENTS.md - Kaggle Ventilator Pressure Prediction

## Project Overview

This is a Kaggle competition project for the Ventilator Pressure Prediction challenge. The goal is to predict airway pressure during mechanical ventilation. The project uses Python with numpy, pandas, and scikit-learn.

## Build/Lint/Test Commands

### Setup
```bash
pip install -r requirements.txt
```

### Running the Project
```bash
python main.py
```

### Testing
- **No formal test suite exists** - This is a Kaggle competition project
- If adding tests, use pytest:
  ```bash
  pytest                          # Run all tests
  pytest path/to/test_file.py     # Run specific test file
  pytest -k test_name            # Run tests matching pattern
  ```

### Linting
- **No linting configuration exists** - Consider adding flake8 or pylint if needed
- Manual checks:
  ```bash
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  ```

### Data
- Input data expected in `../data/` directory relative to notebooks
- Train data: `../data/train.csv`

---

## Code Style Guidelines

### General Principles
- Write clear, readable code suitable for data science/ML experimentation
- Use meaningful variable names that describe the data
- Keep code modular with well-defined functions

### Imports
- Standard library imports first
- Third-party imports second (numpy, pandas, scikit-learn)
- Avoid wildcard imports (`from x import *`)
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
```

### Formatting
- Use 4 spaces for indentation (not tabs)
- Maximum line length: 100 characters (soft guideline)
- Use blank lines to separate logical sections
- Remove commented-out code before committing

### Types
- No type hints required but encouraged for complex functions
- Use numpy arrays and pandas DataFrames appropriately
- Document expected data shapes in comments for ML arrays

### Naming Conventions
- Variables and functions: `snake_case` (e.g., `train_data`, `add_features`)
- Classes: `PascalCase` (e.g., `DataProcessor`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
- Prefix private functions with underscore: `_helper_function()`

### Functions
- Keep functions focused on a single task
- Add docstrings for complex functions:
```python
def add_features(df):
    """Add engineered features to the dataframe.
    
    Args:
        df: Input DataFrame with ventilation data
        
    Returns:
        DataFrame with new features added
    """
```
- Use descriptive function names that explain what they do

### Error Handling
- Use try/except for file I/O operations
- Validate data inputs at function boundaries
- Print informative error messages

### Data Science Specific
- Use in-place operations when memory is a concern (`.inplace=True`)
- Document expected DataFrame columns
- Use groupby operations for breath-level computations
- Handle missing values explicitly (e.g., `.fillna(0)`)

### ML Model Guidelines
- Separate data preprocessing from model training
- Document model hyperparameters
- Use consistent random seeds for reproducibility

### Jupyter Notebooks
- Store exploratory notebooks in `notebooks/` directory
- Use clear section headers
- Clean up temporary variables

---

## Project Structure

```
kaggle-ventilator/
├── main.py                 # Main entry point (currently empty)
├── requirements.txt        # Dependencies
├── README.md              # Project documentation
├── models/                 # Model definitions
│   └── attention_sdpa.py
├── notebooks/             # Jupyter notebooks
│   ├── try.ipynb
│   ├── train_data_generation.py
│   ├── submission_generation.ipynb
│   └── pre_data_generation.ipynb
└── data/                  # Data directory (not tracked in git)
```

---

## Common Tasks

### Generate Training Data
```bash
cd notebooks
python train_data_generation.py
```

### Create Submission
Open and run `notebooks/submission_generation.ipynb`

### Add New Features
Edit the `add_features()` function in `notebooks/train_data_generation.py`

---

## Notes for AI Agents

- This is an experimental ML project, not a production system
- Code quality is secondary to experimentation speed
- Focus on feature engineering and model improvements
- Feel free to add new notebooks for exploration
- Document any significant findings in code comments
