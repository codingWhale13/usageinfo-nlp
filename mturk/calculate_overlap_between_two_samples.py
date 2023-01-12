import pandas as pd


SAMPLES_DIR = 'good_samples'

df1 = pd.read_csv('good_samples/sample_21269605.tsv', sep='\t')
df2 = pd.read_csv(f"{SAMPLES_DIR}/sample_76482724.tsv", sep='\t')

intersection = pd.merge(df1, df2, how='inner', on=['review_id'])

print(intersection)