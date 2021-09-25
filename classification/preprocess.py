########################################
#This script reads data and convert it to a code redable format to be used in next steps
#########################################

import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_pickle("data_all.pkl")

X = []
Y = []
for x in df:
  X.append(x[0])
  Y.append(x[1])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42,stratify=Y)

df_amazon = pd.DataFrame(list(zip(X,Y)),columns=['news','class'])
df_train = pd.DataFrame(list(zip(X_train,Y_train)),columns=['news','class'])
df_test = pd.DataFrame(list(zip(X_test,Y_test)),columns=['news','class'])

df_amazon.to_csv("df.tsv",sep='\t')
df_train.to_csv("df_train.tsv",sep='\t')
df_test.to_csv("df_test.tsv",sep='\t')
