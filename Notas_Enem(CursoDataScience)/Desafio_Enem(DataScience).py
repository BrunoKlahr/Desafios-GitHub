# Desafio AnÃ¡lise de dados - Codenation Curso Data Science
# Bruno Klahr


# Analise de Dados

# Módulos
#import Limpeza as Limpeza

# Bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression


# Lendo os dados originais
dados_train_original = pd.read_csv('C:/Users/bruno/Desktop/Cursos_Online/Codenation/Desafios/Notas_Enem(CursoDataScience)/train.csv') 
dados_test_original = pd.read_csv('C:/Users/bruno/Desktop/Cursos_Online/Codenation/Desafios/Notas_Enem(CursoDataScience)/test.csv') 

## Bloco de Limpeza de dados
#df_train = dados_train_original.copy()
#df_test = dados_test_original.copy()
#df_test, df_train = Limpeza.Limpeza_dados(df_test, df_train)
#df_test.to_csv('C:/Users/bruno/Desktop/Cursos_Online/Codenation/Desafios/Notas_Enem(CursoDataScience)/test_limpo.csv', encoding='utf-8', index=False) 
#df_train.to_csv('C:/Users/bruno/Desktop/Cursos_Online/Codenation/Desafios/Notas_Enem(CursoDataScience)/train_limpo.csv', encoding='utf-8', index=False) 


# Lendo os dados após a limpeza
df_train = pd.read_csv('C:/Users/bruno/Desktop/Cursos_Online/Codenation/Desafios/Notas_Enem(CursoDataScience)/train_limpo.csv') 
df_test= pd.read_csv('C:/Users/bruno/Desktop/Cursos_Online/Codenation/Desafios/Notas_Enem(CursoDataScience)/test_limpo.csv') 


# Exploração dos dados, para verificar quais dados são relevantes para a nota de
# matemática

# Limpando banco:
colunas_removidas = ['NU_NOTA_COMP1','NU_NOTA_COMP2','NU_NOTA_COMP3','NU_NOTA_COMP4',
                     'NU_NOTA_COMP5','Q026','TP_ESCOLA','TP_ANO_CONCLUIU',
                     'TP_ST_CONCLUSAO','NU_IDADE','TP_SEXO','TP_LINGUA',
                     'TP_COR_RACA','Q001','Q002']    
for x in colunas_removidas: 
    df_train.drop(x, axis=1, inplace = True)  # Outra maneira de excluir colunas  
    df_test.drop(x, axis=1, inplace = True)
    
 Matriz de correlações
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#  Matriz de correlações - relacionada com a nota de matemática

#k = 20 #number of variables for heatmap
#cols = corrmat.nlargest(k, 'NU_NOTA_MT')['NU_NOTA_MT'].index
#cm = np.corrcoef(df_train[cols].values.T)
#sns.set(font_scale=1.25)
#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()



#scatter plot
#variavel = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','Q006']
#for var in variavel:
#    data = pd.concat([df_train['NU_NOTA_MT'], df_train[var]], axis=1)
#    data.plot.scatter(x=var, y='NU_NOTA_MT', ylim=(0,1000));  
    
    
# Criando modelo de regressão linear

X_train =  df_train.drop('NU_INSCRICAO', axis=1)
X_train =  X_train.drop('NU_NOTA_MT', axis=1)
Y_train = df_train['NU_NOTA_MT']

X_test =  df_test.drop('NU_INSCRICAO', axis=1)
X_test =  X_test.drop('NU_NOTA_MT', axis=1)


lm = LinearRegression()
lm.fit(X_train,Y_train)

coef = pd.DataFrame(zip(X_train.columns,lm.coef_),columns = ['variaveis','coeficientes'])
#print(coef)

pred_test = pd.DataFrame(lm.predict(X_test), columns = ['NU_NOTA_MT'])

Resposta = pd.merge(df_test['NU_INSCRICAO'], pred_test, how='outer',left_index=True, right_index=True)

Resposta.to_csv('C:/Users/bruno/Desktop/Cursos_Online/Codenation/Desafios/Notas_Enem(CursoDataScience)/answer.csv', encoding='utf-8', index=False) 

    