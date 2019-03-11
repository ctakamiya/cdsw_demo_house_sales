# # Demonstração do CDSW 
#
# ## Configuração inicial do estudo
# 
from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from IPython.display import HTML
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble, tree, linear_model
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

# ## Definindo o formato de uso 'png'
%config InlineBackend.figure_format = 'png' 
%matplotlib inline



# ## Introdução
# ### Base de dados
# Esta demonstração usa o dataset da **Kaggle**, no qual podemos encontrar
# dados referentes a venda de casas no período de Maio de 2014 à Maio de 2015
# no Condado de King, WA EUA.

# ### Análise exploratória.
# Podemos dizer que a qualidade do trabalho de um Cientista de Dados está na 
# "reprodutibilidade" do seu estudo.
#
# "A reprodutibilidade de uma experiência científica é uma das condições que
# permitem incluir no processo de progresso do conhecimento científico 
# as observações realizadas durante a experiência. 
# **Essa condição origina-se no princípio de que não se pode tirar**
# **conclusões senão de um evento bem descrito, que aconteceu várias**
# **vezes, provocado por pessoas distintas. Essa condição permite se **
# **livrar de efeitos aleatórios que podem afetar os resultados, de **
# **erros de julgamento ou de manipulações por parte dos cientistas.**
#   https://pt.wikipedia.org/wiki/Reprodutibilidade

df_kc_house = pd.read_csv("./dataset/kc_house_data.csv")

# ### Total de casas
df_kc_house.id.count()


# ### Retirando um amostra
df_train = df_kc_house.sample(2000)

# ### Limpeza de dados
# O dataset fornecido pelo Kaggle provavelmente passou por um processo
# de Data Cleansing, pois não há dados faltando.
# Nesta demos iremos apenas corrigir alguns nomes de campos.
df_train.rename(columns ={'price': 'SalePrice'}, inplace =True)

# ### Inicio do dataset
df_kc_house.head()

# ### Fim do dataset
df_kc_house.tail()

# ### Variáveis do dataset de treino
# Neste trabalho usaremos todos as colunas (variáveis) do dataset
df_train.columns

# ### Principais estatísticas dos Preços das Casas
# #### Estatística Descritiva
df_train['SalePrice'].describe()

# #### histograma
sns.distplot(df_train['SalePrice'], bins=50, kde=False);

var = 'sqft_living15'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(3,8000000));

var = 'bedrooms'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=3500000);

var = 'grade'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=3500000);

# Matriz de Correlação
corrmat = df_train.corr()
sns.heatmap(corrmat, vmax=.8, square=True);

# Matriz de correlação saleprice
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()



# ### Geoestatística
# Heatmap de preços das casas vendidas pelo Condado de Kinga
mapit = folium.Map(location=[df_train["lat"].mean(), 
                             df_train["long"].mean()], 
                   tiles="OpenStreetMap", zoom_start=9)

mapit.add_child(HeatMap(zip(df_train['lat'], 
                            df_train['long'], 
                            df_train['SalePrice']),
                        radius=10))


# Padronização dos dados para mitigar a assimetria e curtose
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

# Histograma e gráfico e probabilidade normal
sns.distplot(df_train['SalePrice'], fit=norm, bins=50, kde=False);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# Aplicando transformação de log em "SalesPrice"
df_train['SalePrice'] = np.log(df_train['SalePrice'])

# Histograma dos dados transformado e gráfico de probabilidade normal
sns.distplot(df_train['SalePrice'], fit=norm,  bins=50, kde=False);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# Histograma e gráfico e probabilidade normal
sns.distplot(df_train['sqft_living'], fit=norm, bins=50, kde=False);
fig = plt.figure()
res = stats.probplot(df_train['sqft_living'], plot=plt)


# Aplicando transformação de log em "sqft_living"
df_train['sqft_living'] = np.log(df_train['sqft_living'])


# Histograma dos dados transformado e gráfico de probabilidade normal
sns.distplot(df_train['sqft_living'], fit=norm, bins=50, kde=False);
fig = plt.figure()
res = stats.probplot(df_train['sqft_living'], plot=plt)


# Gráfico de Dispersão
plt.scatter(df_train['sqft_living'], df_train['SalePrice']);

# ### Treinando modelo de Regressão Linear
# Nesta parte do estudo iniciaremos o treino do modelo.
# Em nossa abordagem vamos dividir o dataset em dataset de treino 
# e dataset de teste.

df_train = df_kc_house
df_train.rename(columns ={'price': 'SalePrice'}, inplace =True)


Y = df_train.SalePrice.values

feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']
X=df_train[feature_cols]


x_train,x_test,y_train,y_test = train_test_split(X, Y, random_state=3)



regressor = LinearRegression()
regressor.fit(x_train, y_train)

accuracy = regressor.score(x_test, y_test)
"Accuracy: {}%".format(int(round(accuracy * 100)))


# ### Funções para avaliar a precisão do modelo

def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('Raiz do Erro Quadrático Médio: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))
    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Teste")
    get_score(prediction_test, y_tst)
    


ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], 
                                    l1_ratio=[.01, .1, .5, .9, .99],
                                    max_iter=5000).fit(x_train, y_train)

train_test(ENSTest, x_train, x_test, y_train, y_test)

# Média do R$^2$ e o desvio padrão de 5 validações cruzadas
scores = cross_val_score(ENSTest, x_test, y_test, cv=5)
print("Acurácia: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)
train_test(GBest, x_train, x_test, y_train, y_test)



# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(GBest, x_test, y_test, cv=5)
print("Acurácia: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ### Exportando o modelo
# Retire o caracter de comentário da linha para salvar o modelo treinado
#
joblib.dump(GBest, 'gradiente.pkl') 
# gb2 = joblib.load('gradiente.pkl')


# scores = cross_val_score(gb2, x_test, y_test, cv=5)





d ={'bedrooms':3, 'bathrooms':1.75, 'sqft_living':1600, 'sqft_lot':9579, 'floors':1.0,
       'view':0, 'condition':3, 'grade':8, 'sqft_above':1180,
       'sqft_basement':420, 'yr_built':1977, 'yr_renovated':0, 'zipcode':98072, 'lat':47.7662, 'long':-122.159,
       'sqft_living15':1750, 'sqft_lot15':9829}

df = pd.DataFrame(data=d, index=[0])
GBest.predict(df)