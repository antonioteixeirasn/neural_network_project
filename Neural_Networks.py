#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importando as bibliotecas

import numpy as np
import tensorflow as tf
import pandas as pd


# In[2]:


# Importando o dataset

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values# AS 3 primeiras colunas são irrelevantes
y = dataset.iloc[:,-1]

dataset.head()


# In[3]:


# Processamento de dados da columa "Gender"

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X[:, 2] = le.fit_transform(X[:, 2])
# o nº 2 serve para falar qual coluna iremos modificar

'''
Importante: LabelEncoder adiciona um grau de ordem entre as features, porém como
são 2 categorias, poderia ser usado LabelEncoder ou OneHotEncoder. 
Aqui, LabelEncoder é melhor, pois OneHotEncoder adicionaria uma outra coluna.
Com LabelEncoder usamos apenas uma coluna e diferenciamos male de female com 
apenas 1 e 0
'''


# In[4]:


# Processamento de dados da coluna "Geography"

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
# o nº 1 está indicando qual coluna sera modificada para introduzir as dummy variables

X = np.array(ct.fit_transform(X))

'''
Aqui se faz necessário utilizar o OneHotEncoder pois há 3 categorias de países:
França, Espanha e Alemanha. 
Caso usemos LabelEncoder teríamos 1 única coluna com valores de 1 a 3
isso faz com que a rede neural entenda que há uma ordem de grandeza entre as 
features, o que não é verdade.
Por isso, OneHotEncoder é melhor aplicado, criando assim 3 colunas de valores binários.
'''

#Vale lembrar que OneHotEncoder coloca as dummys variables no início da matriz

print(X)


# In[5]:


# Dividindo os dados entre treino e teste

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[6]:


# Efetuando o feature scaling

# Código para efetuar a normalização dos dados antes de treinar a rede neural:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[7]:


# Cosntruindo a Rede Neural

# Iniciando a rede neural
ann = tf.keras.models.Sequential()

# Adicionando a camada de entrada e a primeira camada escondida

ann.add(tf.keras.layers.Dense(units = 10, activation = 'relu'))

# Adicionando a segunda camada de neurônios escondida

ann.add(tf.keras.layers.Dense(units = 10, activation = 'relu'))

# Adicionando a camada de saída

ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# Compilando a rede neural

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

'''
binary_crossentropy é usada quando se tem apenas 2 categorias
para 3 ou mais categorias usa-se category_crossentropy
'''


# In[8]:


# Treinando a rede neural

ann.fit(X_train, y_train, batch_size = 32, epochs = 250)


# In[9]:


# Prevendo os resultados de teste

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)


# In[10]:


# Criando a matriz de confusão

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

