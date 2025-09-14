import re

# Read the file with UTF-8 encoding
with open('scripts/advanced_ml_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace problematic Unicode characters with ASCII equivalents
replacements = {
    '✓': '[OK]',
    '✗': '[ERROR]',
    '➤': '->',
    '✓': '[OK]',
    '✗': '[ERROR]',
    # Japanese characters to English equivalents in comments
    '高度な': 'Advanced',
    '機械学習': 'Machine Learning',
    '特徴量エンジニアリング': 'Feature Engineering',
    '次元削減': 'Dimensionality Reduction',
    'の適用': ' Application',
    '多項式特徴量の作成': 'Polynomial Feature Creation',
    'アンサンブル': 'Ensemble',
    'モデル群': 'Model Zoo',
    'パイプライン': 'Pipeline',
    '最適化': 'Optimization',
    'ハイパーパラメータ': 'Hyperparameter',
    '交差検証': 'Cross Validation',
    'スコア順でソート': 'Sort by Score',
    '最終予測': 'Final Prediction'
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Remove any remaining non-ASCII characters in comments
content = re.sub(r'"""[^"]*[^\x00-\x7F][^"]*"""', '"""Advanced ML Pipeline"""', content)
content = re.sub(r'#[^\n]*[^\x00-\x7F][^\n]*', '# Advanced ML comment', content)

# Write back with UTF-8 encoding
with open('scripts/advanced_ml_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Unicode characters replaced successfully")