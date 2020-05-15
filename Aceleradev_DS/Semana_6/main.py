#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

from loguru import logger


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


fifa = pd.read_csv("fifa.csv")


# In[4]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
# Data visualization
fifa.head()


# In[6]:


# Data shape
fifa.shape


# In[23]:


# Creating an auxiliary data frame with data informations
fifa_columns = pd.DataFrame({'tipo': fifa.dtypes,
                    'mean' :  fifa.mean(),
                    'max' :  fifa.max(),
                    'min' :  fifa.min(),
                    'standard deviation' :  fifa.std(),  
                    'missing' : fifa.isna().sum(),
                    'size' : fifa.shape[0],
                    'unicos': fifa.nunique()})
fifa_columns['percentual [%]'] = 100*round(fifa_columns['missing'] / fifa_columns['size'],5)
fifa_columns


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# __Removendo dados missing__

# In[ ]:


# Removing all the lines with missing values
fifa.dropna(inplace=True)


# In[ ]:


pca = PCA().fit(fifa)      # Fit the PCA (Principal component analysis)
def q1():
    # Retorne aqui o resultado da questão 1.
    explained_variance = pca.explained_variance_ratio_   # Computing the explained variance ratio
    answer = float(round(explained_variance[0],3))       # The explained variance ratio for the first component
    return answer
q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[ ]:


def q2():
    # Retorne aqui o resultado da questão 2.
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)    # Compute the cumulative explained variance ratio
    number_of_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1 # Find the component that the cumulative variance>=95%
    answer = int(number_of_components)
    return answer
q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[ ]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,\
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[ ]:


def q3():
    # Retorne aqui o resultado da questão 3.
    pca2 = PCA(n_components=2).fit(fifa)        # Fit the PCA (Principal component analysis), using 2 components
    x_pca = pca2.components_.dot(x).round(3)    # Calculate the internal product between the pca components and x (round 3 decimals)    
    answer = tuple(x_pca)                       # To convert a array in a tuple
    return answer
q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# __Importing the packages__

# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[ ]:


def q4():
    # Retorne aqui o resultado da questão 4.
    # Defining the rfe_selector with Linear regression, 5 features and step = 1
    rfe_selector = RFE(LinearRegression(), n_features_to_select = 5, step = 1)
    
    # Selection of y and x variables
    y_train = fifa['Overall']                 # The overall feature is the target variable
    x_train = fifa.drop(columns=['Overall'])  # All other features are the independent variables
    
    # Fitting the rfe selector with the training variables
    rfe_selector.fit(x_train, y_train)
    
    # Obtaining the five selected features
    answer = list(x_train.columns[rfe_selector.get_support()])   
    return answer
q4()

