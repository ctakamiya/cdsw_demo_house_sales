import pandas as pd
from sklearn.externals import joblib

gb = joblib.load('gradiente.pkl')

def calcImovel(args):
  d ={'bedrooms': args['bedrooms'], 
      'bathrooms': args['bathrooms'], 
      'sqft_living': args['sqft_living'], 
      'sqft_lot': args['sqft_lot'], 
      'floors': args['floors'],
      'view': args['view'], 
      'condition': args['condition'], 
      'grade': args['grade'], 
      'sqft_above': args['sqft_above'],
      'sqft_basement': args['sqft_basement'], 
      'yr_built': args['yr_built'], 
      'yr_renovated': args['yr_renovated'], 
      'zipcode': args['zipcode'], 
      'lat': args['lat'], 
      'long': args['long'],
      'sqft_living15': args['sqft_living15'], 
      'sqft_lot15': args['sqft_lot15']}
  valor = gb.predict(pd.DataFrame(data=d, index=[0]))
  return valor