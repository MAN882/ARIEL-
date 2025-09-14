# GitHub Setup Instructions

## Manual GitHub Repository Creation

Since GitHub CLI is not installed, please follow these steps to create the GitHub repository manually:

### Step 1: Create Repository on GitHub
1. Go to https://github.com/new
2. Repository name: `ariel-data-challenge-2025`
3. Description: `Ariel Data Challenge 2025 - Multi-detector ensemble solution (CV: 0.000064)`
4. Set as Public or Private (your choice)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Step 2: Connect Local Repository
After creating the GitHub repository, run these commands in the project directory:

```bash
cd "C:\Users\ichry\OneDrive\Desktop\kaggle_competition\ariel_data_challenge_2025"

# Add remote origin
git remote add origin https://github.com/YOUR_USERNAME/ariel-data-challenge-2025.git

# Push to GitHub
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 3: Verify Connection
```bash
git remote -v
```

Should show:
```
origin  https://github.com/YOUR_USERNAME/ariel-data-challenge-2025.git (fetch)
origin  https://github.com/YOUR_USERNAME/ariel-data-challenge-2025.git (push)
```

## Synchronization Workflow

### From Local to Kaggle
1. Make changes locally
2. Commit to Git: `git add . && git commit -m "Update: description"`
3. Push to GitHub: `git push`
4. In Kaggle, use "Add Data" -> "GitHub" to import updated repository
5. Or manually copy notebook content to Kaggle editor

### From Kaggle to Local
1. Export notebook from Kaggle (Download -> .ipynb)
2. Copy to local `notebooks/` directory
3. Commit changes: `git add . && git commit -m "Update from Kaggle"`
4. Push to GitHub: `git push`

## Alternative: GitHub CLI Installation
If you prefer using GitHub CLI, install it from: https://cli.github.com/

Then you can create repositories with:
```bash
gh repo create ariel-data-challenge-2025 --public --description "Ariel Data Challenge 2025 solution"
git remote add origin https://github.com/YOUR_USERNAME/ariel-data-challenge-2025.git
git push -u origin main
```