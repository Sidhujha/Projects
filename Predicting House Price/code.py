import pandas as pd
df=pd.read_csv("house_sales.csv")
print(df)

df.columns
df.info()
df.isnull().sum()
df=df.dropna(axis=0)
df.describe() 

df=df.drop(labels=['id','date','waterfront','view','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15','condition','grade'],axis=1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

temp1 = df.drop(['price'], axis=1)
feature = df['price']

X_train, X_test, y_train, y_test = train_test_split(temp1, feature, test_size=0.30, random_state=52)
predictor_model=LinearRegression()

predictor_model.fit(X_train,y_train)

#predicting the training and testing set and evaluating metrics
sol=predictor_model.predict(X_test)
mae = mean_absolute_error(y_test,sol)
mse = mean_squared_error(y_test,sol)
rmse = mean_squared_error(y_test,sol, squared=False)
r2 = r2_score(y_test,sol)

print(f"""
Mean Absolute error : {mae}
Mean Squared Error : {mse}
Root mean squared error : {rmse}
Rsqaure Score : {r2} 
""")

#for generating more records to improve the accuracy
import numpy as np
import pandas as pd
np.random.seed(42)
num_samples = 20000

bedrooms = np.random.randint(1, 8, num_samples)
bathrooms = np.round(np.random.uniform(1, 5, num_samples), 0)
sqft_living = np.random.randint(500, 10000, num_samples)
sqft_lot = np.random.randint(1000, 15000, num_samples)
floors = np.random.choice([1, 2, 3,4], num_samples)
sqft_above = np.random.randint(500, 8000, num_samples)
sqft_basement = sqft_living - sqft_above
sqft_basement[sqft_basement < 0] = 0  

price = (
    10000 + 50000 * bedrooms +
    30000 * bathrooms +
    150 * sqft_living +
    50 * sqft_lot +
    10000 * floors +
    200 * sqft_above +
    100 * sqft_basement +
    np.random.normal(0, 100000, num_samples)  )
data = {
    'price': price,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft_living': sqft_living,
    'sqft_lot': sqft_lot,
    'floors': floors,
    'sqft_above': sqft_above,
    'sqft_basement': sqft_basement
}

df_trainer = pd.DataFrame(data)
df_trainer.to_csv('synthetic_house_data.csv', index=False)


#to predict the house price
print("HOUSE PRICE PREDICTION MODEL")
print("""
Enter the details of the house :
""")
bedrooms=int(input("No of Bedrooms :"))
bathrooms=int(input("No of Bathrooms :"))
livingarea=int(input("Living area in sqft :"))
lotarea=int(input("Lot area in sqft :"))
floor=int(input("No of floors :"))
sqftabove=int(input("Area above the house :"))
sqftbase=int(input("Area of basement of the house:"))

testing = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [livingarea],
    'sqft_lot': [lotarea],
    'floors': [floor],
    'sqft_above': [sqftabove],
    'sqft_basement': [sqftbase]
})
ans=int(predictor_model.predict(testing))
print()
print("The predicted cost would is :",ans,"US dollars")
print("The r2 score is : ",r2)

