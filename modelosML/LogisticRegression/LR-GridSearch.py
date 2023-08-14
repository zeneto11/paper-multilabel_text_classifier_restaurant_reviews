import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline

start_time = time.time()

# Carregar dataset
df = pd.read_csv(r'C:\VSCODEProjects\artigo-classificador_multilabel\dataset-4000sentencas-multilabel.csv', encoding='utf-8')

# Separando variáveis de entrada (X) e saída (y)
X = df['sentenca']
y = df.drop('sentenca', axis=1)

# Definindo o pipeline com etapas de extração de recursos e seleção de recursos
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('select', SelectPercentile(chi2)),
    ('model', OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=1000)))
])

# Definindo a lista de porcentagens a serem testadas
percentiles = [40, 60, 80, 100]

# Definindo os valores dos hiperparâmetros a serem testados
param_grid = {
    'tfidf__min_df': [1, 3, 5],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'select__percentile': percentiles,
    'model__estimator__C': [0.01, 0.1, 1.0, 10.0],
    'model__estimator__penalty': ['l1', 'l2', 'elasticnet'],
    'model__estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'model__estimator__max_iter': [100, 500, 1000, 1500]
}

# Criando o objeto GridSearchCV
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='f1_macro')

# Realizando a busca pelos melhores hiperparâmetros
grid_search.fit(X, y) # type: ignore

# Exibindo os resultados
with open("resultados-LR_OvA.txt", "w") as f:
    f.write(f"Melhor porcentagem: {grid_search.best_params_['select__percentile']:.2f}\n")
    f.write(f"Melhores hiperparâmetros: {grid_search.best_params_}\n")
    f.write(f"Melhor f1: {grid_search.best_score_:.2f}\n")

print("--- %s seconds ---" % (time.time() - start_time))

print(grid_search.cv_results_['mean_test_score'])
