# Desafio Aceleradev Data Science - Semana 3
# by: Bruno Klahr

# Desafio Coestatistica

# importação de pacotes
import pandas as pd

def main():
    df = pd.read_csv("desafio1.csv")
    # Obtendo os dados do desafio:
    # Moda, mediana, média e desvio padrão dos dados de pontuação de crédito
    # agrupados por estados.
    df_answer = df.groupby("estado_residencia")["pontuacao_credito"] \
            .agg({'mediana': 'median', 'media': 'mean', 'desvio_padrao': 'std'})
    # Resposta para Moda teve que ser realizada separada
    df_mode= df.groupby("estado_residencia")["pontuacao_credito"].agg(pd.Series.mode)
    # Agrupando as respostas
    df_answer['moda'] = df_mode
    # Transpondo o Data Frame
    df_answer = df_answer.transpose()
    # Colocandos os dados de acordo com a resposta
    column_names = ['SC', 'RS', 'PR']
    row_names = ['moda', 'mediana', 'media', 'desvio_padrao']
    df_answer = df_answer.reindex(columns=column_names)
    df_answer = df_answer.reindex(row_names)
    # Salvando em arquivo json
    df_answer.to_json('submission.json')

if __name__ == '__main__':
    main()
    #submission = pd.read_json("submission.json")