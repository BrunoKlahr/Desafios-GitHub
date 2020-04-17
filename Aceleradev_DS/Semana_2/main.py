#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[45]:


import pandas as pd
import numpy as np


# In[46]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[47]:


# Let's go


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[48]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
    #pass
q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[49]:


def q2():
    # Retorne aqui o resultado da questão 2.
    black_friday_F = black_friday[black_friday['Gender'] == 'F']
    number_female_Age26_35 = int(black_friday_F[black_friday_F['Age'] == '26-35'].count()['Age'])
    return number_female_Age26_35
    #pass
q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[50]:


def q3():
    # Retorne aqui o resultado da questão 3.
    number_unique_user = black_friday['User_ID'].nunique()
    return number_unique_user
    #pass
q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[51]:


def q4():
    # Retorne aqui o resultado da questão 4.
    type_datas_unique = black_friday.dtypes.nunique()
    return type_datas_unique
    #pass
q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[52]:


def q5():
    # Retorne aqui o resultado da questão 5.
    black_friday_dropna = black_friday.dropna()
    n_not_na = black_friday_dropna.shape[0]
    total = black_friday.shape[0]
    perc_null_value = (total-n_not_na)/total
    return perc_null_value
    #pass
q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[53]:


def q6():
    # Retorne aqui o resultado da questão 6.
    df_dados_faltantes = pd.DataFrame({'colunas': black_friday.columns,
                                       'percentual_faltante': black_friday.isna().sum()})
    max_dados_faltantes_coluna = df_dados_faltantes['percentual_faltante'].max()  
    return int(max_dados_faltantes_coluna)
    #pass
q6()    


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[54]:


def q7():
    # Retorne aqui o resultado da questão 7.
    black_friday_dropna = black_friday.dropna()
    most_value_freq =  black_friday_dropna['Product_Category_3'].value_counts().idxmax()

    return most_value_freq
    #pass
q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[89]:


def q8():
    # Retorne aqui o resultado da questão 8.
    # Normalização da coluna Purchase utilizando Min - Max
        # Normalized_data = (data - minimuum)/(maximuum - minuimuum)
    maximuum= black_friday['Purchase'].max()
    minimuum = black_friday['Purchase'].min()
    black_friday_aux = (black_friday['Purchase'] - minimuum)/(maximuum-minimuum)
    
    mean_normalized = black_friday_aux.mean()
    return float(mean_normalized)
    #pass
q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[87]:


def q9():
    # Retorne aqui o resultado da questão 9.
        #Padronização pelo z-score
        # Padronized_data = (data - mean)/standard_deviation
    # Padronização
    mean = black_friday['Purchase'].mean()
    std= black_friday['Purchase'].std()
    black_friday_aux = (black_friday['Purchase'] - mean)/std
    
    # Número de ocorrência entre -1 e 1
    return int(black_friday_aux[black_friday_aux < 1][-1 < black_friday_aux].count())
    #pass
q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[40]:


def q10():
    # Retorne aqui o resultado da questão 10.
    
    # Selecionar apenas os registros em que a variável Product_Category_2 é null
    df_aux = black_friday[black_friday['Product_Category_2'].isnull()]
    # Número de valores não nulos da coluna Product_Category_2 (Deve ser zero)
    num_valor_not_null_Cat2 = df_aux ['Product_Category_2'].count()
    assert num_valor_not_null_Cat2 == 0
    # Número de valores não nulos da coluna Product_Category_3
    num_valor_not_null_Cat3 = df_aux ['Product_Category_3'].count()

    '''Se o valor da variável num_valor_not_null_Cat2 for igual a variável num_valor_not_null_Cat2, significa que para todos
    os valores nulos da coluna Product_Category_2 o valor também é nulo para a variável Product_Category_3.'''
    
    if num_valor_not_null_Cat2 == num_valor_not_null_Cat3:
        return True
    else:
        return False
    #pass
q10()


# In[ ]:




