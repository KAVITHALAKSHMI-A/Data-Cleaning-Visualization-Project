
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: Generate Raw Messy Dataset
# ─────────────────────────────────────────────
np.random.seed(42)
n = 300

categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Furniture']
regions    = ['North', 'South', 'East', 'West']
dates      = [datetime(2024, 1, 1) + timedelta(days=i) for i in
              np.random.randint(0, 365, n).tolist()]

raw_data = pd.DataFrame({
    'date':     dates,
    'category': np.random.choice(categories, n),
    'region':   np.random.choice(regions, n),
    'sales':    np.random.normal(500, 150, n),
    'units':    np.random.randint(1, 50, n),
    'discount': np.random.uniform(0, 0.4, n),
    'rating':   np.random.uniform(1, 5, n),
})

# Inject problems ───────────────────────────
# 1. Missing values (10%)
for col in ['sales', 'rating', 'discount']:
    mask = np.random.random(n) < 0.10
    raw_data.loc[mask, col] = np.nan

# 2. Outliers (5 extreme values)
raw_data.loc[np.random.choice(n, 5, replace=False), 'sales'] = \
    np.random.choice([2500, 3000, -200, 2800, 3500], 5)

# 3. Duplicates (15 duplicate rows)
dupes = raw_data.sample(15, random_state=1)
raw_data = pd.concat([raw_data, dupes], ignore_index=True)

raw_data.to_csv('/home/claude/data_project/raw_data.csv', index=False)
print(f"Raw dataset shape: {raw_data.shape}")
print(f"Missing values:\n{raw_data.isnull().sum()}\n")
print(f"Duplicates: {raw_data.duplicated().sum()}")

# ─────────────────────────────────────────────
# STEP 2: Data Cleaning
# ─────────────────────────────────────────────
df = raw_data.copy()

# 2a. Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {before - len(df)}")

# 2b. Fill missing values
df['sales'].fillna(df['sales'].median(), inplace=True)
df['rating'].fillna(df['rating'].mean(), inplace=True)
df['discount'].fillna(0, inplace=True)

# 2c. Handle outliers (IQR method — cap, don't drop)
Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers_count = ((df['sales'] < lower) | (df['sales'] > upper)).sum()
df['sales'] = df['sales'].clip(lower, upper)
print(f"Outliers capped: {outliers_count}")

# 2d. Feature engineering
df['date']    = pd.to_datetime(df['date'])
df['month']   = df['date'].dt.month_name()
df['month_n'] = df['date'].dt.month
df['revenue'] = df['sales'] * df['units'] * (1 - df['discount'])

df.to_csv('/home/claude/data_project/clean_data.csv', index=False)
print(f"\nClean dataset shape: {df.shape}")

# ─────────────────────────────────────────────
# STEP 3: Exploratory Analysis
# ─────────────────────────────────────────────
print("\n=== Key Statistics ===")
print(df[['sales', 'units', 'revenue', 'rating']].describe().round(2))

cat_summary = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
monthly     = df.groupby('month_n')['revenue'].sum()
print(f"\nTop category: {cat_summary.index[0]} (${cat_summary.iloc[0]:,.0f})")
print(f"Best month: Month {monthly.idxmax()} (${monthly.max():,.0f})")

# ─────────────────────────────────────────────
# STEP 4: Visualization Dashboard
# ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#FAFAF8',
    'axes.facecolor': '#FAFAF8',
})

colors = ['#534AB7', '#1D9E75', '#D85A30', '#BA7517', '#A32D2D']

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Sales Data Dashboard — 2024', fontsize=22, fontweight='bold',
             y=0.98, color='#2C2C2A')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

# Chart 1: Revenue by Category (bar)
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(cat_summary.index, cat_summary.values / 1000,
               color=colors, edgecolor='white', linewidth=0.8, width=0.65)
ax1.set_title('Revenue by Category', fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel('Revenue ($ thousands)', fontsize=11)
ax1.tick_params(axis='x', rotation=20)
for bar, val in zip(bars, cat_summary.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'${val/1000:.0f}k', ha='center', va='bottom', fontsize=9,
             fontweight='bold', color='#2C2C2A')

# Chart 2: Monthly Revenue Trend (line)
ax2 = fig.add_subplot(gs[0, 1])
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
ax2.plot(monthly.index, monthly.values / 1000, color='#534AB7',
         linewidth=2.5, marker='o', markersize=6, markerfacecolor='white',
         markeredgewidth=2)
ax2.fill_between(monthly.index, monthly.values / 1000,
                 alpha=0.12, color='#534AB7')
ax2.set_xticks(monthly.index)
ax2.set_xticklabels([month_labels[i-1] for i in monthly.index],
                    rotation=30, fontsize=9)
ax2.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold', pad=12)
ax2.set_ylabel('Revenue ($ thousands)', fontsize=11)

# Chart 3: Sales Distribution Box Plot (before vs after cleaning)
ax3 = fig.add_subplot(gs[1, 0])
raw_sales_clean = raw_data['sales'].dropna()
box_data  = [raw_sales_clean.values, df['sales'].values]
bp = ax3.boxplot(box_data, patch_artist=True, widths=0.55,
                 medianprops=dict(color='white', linewidth=2.5))
for patch, c in zip(bp['boxes'], ['#D85A30', '#1D9E75']):
    patch.set(facecolor=c, alpha=0.75, linewidth=0)
ax3.set_xticklabels(['Before Cleaning', 'After Cleaning'], fontsize=11)
ax3.set_title('Outlier Removal — Sales Distribution', fontsize=14,
              fontweight='bold', pad=12)
ax3.set_ylabel('Sales ($)', fontsize=11)

# Chart 4: Correlation Heatmap
ax4 = fig.add_subplot(gs[1, 1])
corr_cols = ['sales', 'units', 'discount', 'rating', 'revenue']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax4, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, vmin=-1, vmax=1,
            annot_kws={'size': 10, 'weight': 'bold'},
            linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8})
ax4.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
ax4.tick_params(axis='x', rotation=30)
ax4.tick_params(axis='y', rotation=0)

plt.savefig('/home/claude/data_project/dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#FAFAF8')
plt.close()
print("\nDashboard saved to dashboard.png ✓")
