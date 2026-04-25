import pandas as pd

df = pd.read_csv("dataframe_imputado.csv", sep=';')
df = df.dropna()
df.to_csv("dataframe_imputado2.csv", sep=';', index=False, encoding='latin-1', errors='ignore')