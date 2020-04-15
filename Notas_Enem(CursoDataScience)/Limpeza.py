# Desafio AnÃ¡lise de dados - Codenation Curso Data Science
# Bruno Klahr


# Analise de Dados

# Bibliotecas
import pandas as pd

def Limpeza_dados(df_test, df_train):
    # Limpeza inicial do banco de dados
    
    # Criar coluna das notas de matemática para o df_test
    df_test['NU_NOTA_MT'] = 0
    
    # Remover as colunas do df_train que não existem no df_test
    colunas_train = df_train.columns  # Colunas do data frame de treino
    colunas_test = df_test.columns    # Colunas do data frame de teste
    
    for i in range(len(colunas_train)):
        for j in range(len(colunas_test)): 
            if (colunas_train[i] == colunas_test[j]):
                remove_coluna = False
                break
            else:
                remove_coluna = True
        if remove_coluna:
            del df_train[colunas_train[i]]  # Comando para excluir colunas de DF
    
    # Remover colunas visualmente irrelevantes para a nota de matemática
    colunas_removidas = ['CO_UF_RESIDENCIA','SG_UF_RESIDENCIA','TP_NACIONALIDADE',
                         'TP_ENSINO','TP_DEPENDENCIA_ADM_ESC','CO_PROVA_CN','CO_PROVA_CH',
                         'CO_PROVA_LC','CO_PROVA_MT','TP_STATUS_REDACAO','Q027','IN_TREINEIRO',
                         'IN_BAIXA_VISAO', 'IN_CEGUEIRA','IN_SURDEZ','IN_DISLEXIA',
                         'IN_DISCALCULIA','IN_SABATISTA','IN_GESTANTE','IN_IDOSO',
                         'TP_PRESENCA_CN','TP_PRESENCA_CH','TP_PRESENCA_LC']    
    for x in colunas_removidas: 
        df_train.drop(x, axis=1, inplace = True)  # Outra maneira de excluir colunas  
        df_test.drop(x, axis=1, inplace = True)
    
    # Dados faltantes do Data Frame de treino
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data_train = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    # Excluindo dados faltantes
    colunas = (missing_data_train[missing_data_train['Total'] > 1]).index
    for item in colunas:
        aux = df_train[item].isna()
        for i in aux.index:
            if (aux[i]):
                df_train = df_train.drop(i)
            
    #print(df_train.isnull().sum().max())
    
    # Dados faltantes do Data Frame de teste
    total = df_test.isnull().sum().sort_values(ascending=False)
    percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
    missing_data_test = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    # Excluindo dados faltantes
    colunas = (missing_data_test[missing_data_test['Total'] > 1]).index
    for item in colunas:
        aux = df_test[item].isna()
        for i in aux.index:
            if (aux[i]):
                df_test = df_test.drop(i)
            
    #print(df_test.isnull().sum().max())
    
    # Substituir caracteres por dados numéricos
    df_train['TP_SEXO'] = df_train['TP_SEXO'].where(df_train['TP_SEXO'] != 'M', 1.00)
    df_train['TP_SEXO'] = df_train['TP_SEXO'].where(df_train['TP_SEXO'] != 'F', 2.00)
    
    df_train['Q001'] = df_train['Q001'].where(df_train['Q001'] != 'A', 1)
    df_train['Q001'] = df_train['Q001'].where(df_train['Q001'] != 'B', 2)
    df_train['Q001'] = df_train['Q001'].where(df_train['Q001'] != 'C', 3)
    df_train['Q001'] = df_train['Q001'].where(df_train['Q001'] != 'D', 4)
    df_train['Q001'] = df_train['Q001'].where(df_train['Q001'] != 'E', 5)
    df_train['Q001'] = df_train['Q001'].where(df_train['Q001'] != 'F', 6)
    df_train['Q001'] = df_train['Q001'].where(df_train['Q001'] != 'G', 7)
    df_train['Q001'] = df_train['Q001'].where(df_train['Q001'] != 'H', 8)
    
    df_train['Q002'] = df_train['Q002'].where(df_train['Q002'] != 'A', 1)
    df_train['Q002'] = df_train['Q002'].where(df_train['Q002'] != 'B', 2)
    df_train['Q002'] = df_train['Q002'].where(df_train['Q002'] != 'C', 3)
    df_train['Q002'] = df_train['Q002'].where(df_train['Q002'] != 'D', 4)
    df_train['Q002'] = df_train['Q002'].where(df_train['Q002'] != 'E', 5)
    df_train['Q002'] = df_train['Q002'].where(df_train['Q002'] != 'F', 6)
    df_train['Q002'] = df_train['Q002'].where(df_train['Q002'] != 'G', 7)
    df_train['Q002'] = df_train['Q002'].where(df_train['Q002'] != 'H', 8)
    
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'A', 1)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'B', 2)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'C', 3)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'D', 4)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'E', 5)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'F', 6)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'G', 7)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'H', 8)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'I', 9)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'J', 10)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'K', 11)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'L', 12)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'M', 13)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'N', 14)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'O', 15)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'P', 16)
    df_train['Q006'] = df_train['Q006'].where(df_train['Q006'] != 'Q', 17)
    
    df_train['Q024'] = df_train['Q024'].where(df_train['Q024'] != 'A', 1)
    df_train['Q024'] = df_train['Q024'].where(df_train['Q024'] != 'B', 2)
    df_train['Q024'] = df_train['Q024'].where(df_train['Q024'] != 'C', 3)
    df_train['Q024'] = df_train['Q024'].where(df_train['Q024'] != 'D', 4)
    df_train['Q024'] = df_train['Q024'].where(df_train['Q024'] != 'E', 5)
    
    df_train['Q025'] = df_train['Q025'].where(df_train['Q025'] != 'A', 1)
    df_train['Q025'] = df_train['Q025'].where(df_train['Q025'] != 'B', 2)

    df_train['Q026'] = df_train['Q026'].where(df_train['Q026'] != 'A', 1)
    df_train['Q026'] = df_train['Q026'].where(df_train['Q026'] != 'B', 2)
    df_train['Q026'] = df_train['Q026'].where(df_train['Q026'] != 'C', 3)
    
    df_train['Q047'] = df_train['Q047'].where(df_train['Q047'] != 'A', 1)
    df_train['Q047'] = df_train['Q047'].where(df_train['Q047'] != 'B', 2)
    df_train['Q047'] = df_train['Q047'].where(df_train['Q047'] != 'C', 3)
    df_train['Q047'] = df_train['Q047'].where(df_train['Q047'] != 'D', 4)
    df_train['Q047'] = df_train['Q047'].where(df_train['Q047'] != 'E', 5)
    
    
    
    df_test['TP_SEXO'] = df_test['TP_SEXO'].where(df_test['TP_SEXO'] != 'M', 1.00)
    df_test['TP_SEXO'] = df_test['TP_SEXO'].where(df_test['TP_SEXO'] != 'F', 2.00)
    df_test['Q001'] = df_test['Q001'].where(df_test['Q001'] != 'A', 1)
    df_test['Q001'] = df_test['Q001'].where(df_test['Q001'] != 'B', 2)
    df_test['Q001'] = df_test['Q001'].where(df_test['Q001'] != 'C', 3)
    df_test['Q001'] = df_test['Q001'].where(df_test['Q001'] != 'D', 4)
    df_test['Q001'] = df_test['Q001'].where(df_test['Q001'] != 'E', 5)
    df_test['Q001'] = df_test['Q001'].where(df_test['Q001'] != 'F', 6)
    df_test['Q001'] = df_test['Q001'].where(df_test['Q001'] != 'G', 7)
    df_test['Q001'] = df_test['Q001'].where(df_test['Q001'] != 'H', 8)
    df_test['Q002'] = df_test['Q002'].where(df_test['Q002'] != 'A', 1)
    df_test['Q002'] = df_test['Q002'].where(df_test['Q002'] != 'B', 2)
    df_test['Q002'] = df_test['Q002'].where(df_test['Q002'] != 'C', 3)
    df_test['Q002'] = df_test['Q002'].where(df_test['Q002'] != 'D', 4)
    df_test['Q002'] = df_test['Q002'].where(df_test['Q002'] != 'E', 5)
    df_test['Q002'] = df_test['Q002'].where(df_test['Q002'] != 'F', 6)
    df_test['Q002'] = df_test['Q002'].where(df_test['Q002'] != 'G', 7)
    df_test['Q002'] = df_test['Q002'].where(df_test['Q002'] != 'H', 8)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'A', 1)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'B', 2)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'C', 3)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'D', 4)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'E', 5)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'F', 6)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'G', 7)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'H', 8)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'I', 9)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'J', 10)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'K', 11)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'L', 12)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'M', 13)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'N', 14)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'O', 15)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'P', 16)
    df_test['Q006'] = df_test['Q006'].where(df_test['Q006'] != 'Q', 17)
    df_test['Q024'] = df_test['Q024'].where(df_test['Q024'] != 'A', 1)
    df_test['Q024'] = df_test['Q024'].where(df_test['Q024'] != 'B', 2)
    df_test['Q024'] = df_test['Q024'].where(df_test['Q024'] != 'C', 3)
    df_test['Q024'] = df_test['Q024'].where(df_test['Q024'] != 'D', 4)
    df_test['Q024'] = df_test['Q024'].where(df_test['Q024'] != 'E', 5)
    df_test['Q025'] = df_test['Q025'].where(df_test['Q025'] != 'A', 1)
    df_test['Q025'] = df_test['Q025'].where(df_test['Q025'] != 'B', 2)
    df_test['Q026'] = df_test['Q026'].where(df_test['Q026'] != 'A', 1)
    df_test['Q026'] = df_test['Q026'].where(df_test['Q026'] != 'B', 2)
    df_test['Q026'] = df_test['Q026'].where(df_test['Q026'] != 'C', 3)
    df_test['Q047'] = df_test['Q047'].where(df_test['Q047'] != 'A', 1)
    df_test['Q047'] = df_test['Q047'].where(df_test['Q047'] != 'B', 2)
    df_test['Q047'] = df_test['Q047'].where(df_test['Q047'] != 'C', 3)
    df_test['Q047'] = df_test['Q047'].where(df_test['Q047'] != 'D', 4)
    df_test['Q047'] = df_test['Q047'].where(df_test['Q047'] != 'E', 5)



    
    return df_test, df_train