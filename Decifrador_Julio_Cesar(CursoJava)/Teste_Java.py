# Teste Curso Java - Codination
# Bruno Klahr


import requests # Biblioteca de Requests
import json
import hashlib



def decifrador_JC(texto, chave=0):
    resposta = ''
    for x in texto:
        if (ord(x) >= 97 and ord(x) <= 122):
            numero_char = ord(x)- 96 - chave
            if (numero_char<0):
                resposta +=chr(122 + numero_char)
            else:
                resposta += chr(96 + numero_char)
        else:
            resposta += x
    return resposta
        


GETurl = 'https://api.codenation.dev/v1/challenge/dev-ps/generate-data?token='
POSTurl = 'https://api.codenation.dev/v1/challenge/dev-ps/submit-solution?token='
token = '7302e4d2b951decdc4c499b14470676b3d04ea7b'
dado = requests.get(GETurl+token)
dado_dict = dado.json()
    
# Decodificando o texto criptografado
chave = dado_dict['numero_casas']
texto_coded = dado_dict['cifrado']
texto_decoded = decifrador_JC(texto_coded,chave)
#print(texto_decoded)
dado_dict['decifrado'] = texto_decoded
#print(dado_dict)

# Criando resumo criptografado e escrevendo o arquivo .json
with open('answer.json', 'w') as file: # abrindo arquivo
    encoding = file.encoding           # Pegando o encoding
    resumo = hashlib.sha1(texto_decoded.encode(encoding)).hexdigest() # Codificando
    dado_dict['resumo_criptografico'] = resumo
    #print(dado_dict)
    
    # Escrevendo no arquivo .json
    json.dump(dado_dict, file)
    file.close()
    
# Enviando resposta
answer = {'answer': open('answer.json', 'rb')}
r = requests.post(POSTurl+token, files=answer)
print(r.text)

