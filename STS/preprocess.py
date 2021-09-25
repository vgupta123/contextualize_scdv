import pandas as pd
df_dev = pd.read_csv("sts-dev.csv")
df_train = pd.read_csv("sts-train.csv")
df_test = pd.read_csv("sts-test.csv")
df = df_train + df_test + df_dev 

X = []
Y = []
for idx, row in df.iterrows():
  X.append(row[5])
  Y.append(row[6])

df = pd.DataFrame(list(zip(X,Y)),columns=['news','class'])
df.to_csv("df.tsv",sep='\t')
