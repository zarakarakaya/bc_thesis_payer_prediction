import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data import load_data
from src.config import load_config
from src.explainability.helper import get_samples
import torch
from pathlib import Path

folder = "sampler"
out_dir = Path("results") / folder / "best" / "model.pt"
cfg = load_config("configs/best.yaml")

model = torch.load(out_dir, weights_only=False)
model.eval()

groups = get_samples(cfg, model)


X, y, feature_names = load_data(cfg.data.path)
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

df = pd.DataFrame(X_test, columns=feature_names)
df['group'] = 'TN'
df.loc[groups['TP'], 'group'] = 'TP'
df.loc[groups['FP'], 'group'] = 'FP'
df.loc[groups['FN'], 'group'] = 'FN'



features_to_plot = ['ecpm', 'session_count', 'time_played', 
                    'jobs_completed']

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, feature in zip(axes, features_to_plot):
    sns.violinplot(data=df, x='group', y=feature, 
                   order=['TP', 'FP', 'TN', 'FN'], 
                   palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], ax=ax)
    ax.set_title(feature)
    ax.set_xlabel('')
for feature in features_to_plot:
    print(f"\n{feature}:")
    print(df.groupby('group')[feature].agg(['mean', 'std']).round(2))

plt.suptitle('Feature distributions: TP, FP, TN, FN')
plt.tight_layout()
plt.savefig('results/groups_violin.png', dpi=150, bbox_inches='tight')
plt.show()