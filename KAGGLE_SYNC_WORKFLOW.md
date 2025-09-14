# Kaggle-Local Synchronization Workflow

## Overview
This document outlines the workflow for synchronizing your local development environment with Kaggle notebooks using Git + GitHub.

## Initial Setup (One-time)
1. **Local Git Repository**: ✓ Already initialized
2. **GitHub Repository**: Follow instructions in `GITHUB_SETUP.md`
3. **Kaggle Account**: Ensure you have access to kaggle.com

## Development Workflows

### Workflow 1: Local Development → Kaggle
**Best for**: Major development, feature engineering, model experiments

1. **Develop locally**
   ```bash
   # Make changes to scripts, notebooks, or data
   # Test your code locally
   python scripts/notebook_compatible_pipeline.py
   ```

2. **Commit changes**
   ```bash
   git add .
   git commit -m "Add: new feature engineering approach"
   git push
   ```

3. **Import to Kaggle**
   - Option A: Use Kaggle's GitHub integration
     - In Kaggle notebook: "File" → "Import GitHub Repository"
     - Enter: `YOUR_USERNAME/ariel-data-challenge-2025`
   
   - Option B: Manual notebook transfer
     - Open `notebooks/ariel_complete_submission.ipynb` locally
     - Copy content to Kaggle notebook editor

### Workflow 2: Kaggle Development → Local
**Best for**: Quick experiments, competition-specific adjustments

1. **Develop in Kaggle**
   - Use Kaggle's built-in editor
   - Access to full competition dataset
   - GPU/TPU resources available

2. **Export from Kaggle**
   - Kaggle notebook: "File" → "Download" → "Download .ipynb"
   - Or copy code cells manually

3. **Update local repository**
   ```bash
   # Replace local notebook with Kaggle version
   cp ~/Downloads/notebook.ipynb notebooks/ariel_complete_submission.ipynb
   
   # Commit changes
   git add .
   git commit -m "Update: improvements from Kaggle experiments"
   git push
   ```

### Workflow 3: Bidirectional Sync
**Best for**: Collaborative development, backup strategy

1. **Always start with pull**
   ```bash
   git pull origin main
   ```

2. **Make changes** (local or Kaggle)

3. **Commit and push frequently**
   ```bash
   git add .
   git commit -m "Progress: incremental improvement"
   git push
   ```

## File Organization

### Core Files for Kaggle
- `notebooks/ariel_complete_submission.ipynb` - Main submission notebook
- `scripts/kaggle_offline_code.py` - Offline-compatible version
- `scripts/notebook_compatible_pipeline.py` - Standalone pipeline

### Local Development Files
- `scripts/advanced_ml_pipeline.py` - Full pipeline with all features
- `results/` - Analysis results and visualizations
- `data/` - Competition data (excluded from Git)

## Kaggle-Specific Considerations

### Data Handling
```python
# In Kaggle notebook, paths are:
'/kaggle/input/ariel-data-challenge-2025/'

# In local environment, paths are:
'data/'
```

### Resource Management
```python
# Kaggle environment
n_jobs = -1  # Use all available cores

# Local development (adjust based on your machine)
n_jobs = 4   # Conservative setting
```

### Output Files
```python
# Kaggle submission requirement
submission.to_csv('submission.csv', index=False)

# Local development
submission.to_csv('submissions/latest_submission.csv', index=False)
```

## Best Practices

### 1. Version Control
- Commit frequently with descriptive messages
- Use branches for experimental features
- Tag successful submissions

### 2. Code Compatibility
- Test code in both environments
- Use relative imports where possible
- Handle path differences gracefully

### 3. Documentation
- Update README files with significant changes
- Document model performance improvements
- Keep track of successful parameter combinations

### 4. Backup Strategy
- Push to GitHub before major experiments
- Keep successful submissions in separate files
- Export Kaggle notebooks regularly

## Troubleshooting

### Common Issues
1. **Path differences**: Use `pathlib` for cross-platform compatibility
2. **Environment differences**: Test in both local and Kaggle environments
3. **Resource limits**: Adjust `n_jobs` and model complexity for each environment
4. **Data access**: Ensure data paths are correct for each environment

### Debugging Commands
```bash
# Check repository status
git status

# View commit history
git log --oneline

# Check remote connection
git remote -v

# View differences
git diff
```

## Success Metrics
- ✓ Code runs in both environments
- ✓ Consistent results between local and Kaggle
- ✓ Regular commits with meaningful messages
- ✓ Successful Kaggle submissions
- ✓ Backup of all important work