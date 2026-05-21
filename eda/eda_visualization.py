import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


df = pd.read_csv('ted_clean_100k.csv')
print(f"Loaded: {len(df):,} records")

# VALUE_EURO distribution (linear, log, boxplot)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(df['VALUE_EURO'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('VALUE_EURO (EUR)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('VALUE_EURO Distribution')

axes[1].hist(df['VALUE_EURO'].dropna(), bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('VALUE_EURO (EUR)')
axes[1].set_ylabel('Frequency (log)')
axes[1].set_title('VALUE_EURO Distribution (log scale)')
axes[1].set_yscale('log')

axes[2].boxplot(df['VALUE_EURO'].dropna(), vert=True)
axes[2].set_ylabel('VALUE_EURO (EUR)')
axes[2].set_title('Outlier Detection (Boxplot)')

plt.tight_layout()
plt.savefig('eda_value_euro_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_value_euro_distribution.png")

# Outlier statistics via IQR
Q1 = df['VALUE_EURO'].quantile(0.25)
Q3 = df['VALUE_EURO'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold_high = Q3 + 1.5 * IQR
outliers = df[df['VALUE_EURO'] > outlier_threshold_high]
print(f"Outliers (IQR rule): {len(outliers):,} ({len(outliers)/len(df)*100:.1f}%)")

# Top 20 country distribution
country_counts = df['ISO_COUNTRY_CODE'].value_counts().head(20)
plt.figure(figsize=(14, 6))
bars = plt.bar(range(len(country_counts)), country_counts.values,
               color='teal', edgecolor='black', alpha=0.7)
plt.xticks(range(len(country_counts)), country_counts.index, rotation=45, ha='right')
plt.xlabel('Country Code')
plt.ylabel('Record Count')
plt.title('Top 20 Country Distribution')
plt.grid(axis='y', alpha=0.3)
bars[0].set_color('darkgreen')
bars[-1].set_color('darkred')
plt.tight_layout()
plt.savefig('eda_country_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_country_distribution.png")

# Top 15 CPV distribution
cpv_counts = df['CPV'].value_counts().head(15)
plt.figure(figsize=(14, 6))
plt.barh(range(len(cpv_counts)), cpv_counts.values, color='purple',
         edgecolor='black', alpha=0.7)
plt.yticks(range(len(cpv_counts)), cpv_counts.index)
plt.xlabel('Record Count')
plt.ylabel('CPV Code')
plt.title('Top 15 CPV Categories')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_cpv_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_cpv_distribution.png")

# COMBINED_TEXT length distribution
df['text_length'] = df['COMBINED_TEXT'].str.len()
df['word_count'] = df['COMBINED_TEXT'].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['text_length'], bins=50, color='orange', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Character count')
axes[0].set_ylabel('Frequency')
axes[0].set_title('COMBINED_TEXT - Character length')
axes[0].axvline(df['text_length'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["text_length"].mean():.0f}')
axes[0].legend()

axes[1].hist(df['word_count'], bins=50, color='green', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Word count')
axes[1].set_ylabel('Frequency')
axes[1].set_title('COMBINED_TEXT - Word count')
axes[1].axvline(df['word_count'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["word_count"].mean():.0f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('eda_text_length_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_text_length_distribution.png")

# Missing values
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

if len(missing_pct) > 0:
    plt.figure(figsize=(10, max(6, len(missing_pct) * 0.4)))
    plt.barh(range(len(missing_pct)), missing_pct.values,
             color='crimson', edgecolor='black', alpha=0.7)
    plt.yticks(range(len(missing_pct)), missing_pct.index)
    plt.xlabel('Missing percentage (%)')
    plt.title('Missing values after preprocessing')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_missing_values.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: eda_missing_values.png")
