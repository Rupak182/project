import pandas as pd

# Load default datasets
# GossipCop (~22k rows) - Huge boost for accuracy
gossipcop_fake = pd.read_csv('data/gossipcop_fake.csv')
gossipcop_fake['label'] = 1

gossipcop_real = pd.read_csv('data/gossipcop_real.csv')
gossipcop_real['label'] = 0

# Politifact (~1k rows)
politifact_fake = pd.read_csv('data/politifact_fake.csv')
politifact_fake['label'] = 1

politifact_real = pd.read_csv('data/politifact_real.csv')
politifact_real['label'] = 0

# Combine all
combined_df = pd.concat([gossipcop_fake, gossipcop_real, politifact_fake, politifact_real], ignore_index=True)

# Use title as text
combined_df['text'] = combined_df['title']

# Save
combined_df.to_csv('data/combined_news.csv', index=False)

print(f"Combined data saved. Total rows: {len(combined_df)}")
print(f"Labels: {combined_df['label'].value_counts().to_dict()}")
