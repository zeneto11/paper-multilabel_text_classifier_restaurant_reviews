# Importanto bibliotecas
import re
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Carregar dataset
df = pd.read_csv('datasets/dataset-4000sentencas-multilabel.csv', encoding='utf-8') # type: ignore

'''# Converter todas a letras para minúsculas
df['sentenca'] = df['sentenca'].str.lower()'''

'''# Remover espaços extras
df['sentenca'] = df['sentenca'].apply(lambda x: x.strip())'''

'''# Remover pontuação
df['sentenca'] = df['sentenca'].apply(lambda x: re.sub(r'[^\w\s]', '', x))'''

'''# Remover stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))
df['sentenca'] = df['sentenca'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words])) # type: ignore'''

'''# Realizar Lemmatization
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def lemmatize_words(sentence):
    words = word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)
df['sentenca'] = df['sentenca'].apply(lemmatize_words)'''

# Realizar Stemming
from nltk.stem import PorterStemmer
nltk.download('punkt')
stemmer = PorterStemmer()
def stem_words(sentence):
    words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)
df['sentenca'] = df['sentenca'].apply(stem_words)

'''# Utilizar POS tagging do spaCy - remove todas as palavras que não são substantivos, verbos ou adjetivos 
nlp = spacy.load('pt_core_news_sm')
df['sentenca'] = df['sentenca'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_stop and token.pos_ in 
                                                          ['NOUN', 'VERB','ADJ', 'AUX', 'PROP']]))'''

# Definir a semente para garantir a mesma divisão aleatória em cada execução
np.random.seed(123)

# Criar um índice aleatório das linhas do DataFrame
indice_aleatorio = np.random.permutation(df.index)

# dividir o índice em 5 partes iguais
partes_aleatorias = np.array_split(indice_aleatorio, 5)

# Exportar 5 DataFrames correspondentes a cada parte aleatória
for i, parte in enumerate(partes_aleatorias):
    df_parte = df.iloc[parte]
    df_parte.to_csv(f'datasets/dataset-fold_{i+1}-multilabel.csv', encoding = 'utf-8', index=False)