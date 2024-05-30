import pandas as pd
df=pd.read_csv("insurance-ltv-decrease.csv")

df.loc[13933].income
cond=df['income'] == '<=2L'
df=df[cond==False]
df['income']=df['income'].map({'2L-5L':'2L-5L','5L-10L':'5L-10L','More than 10L': '10L-25L'})
df[['lower_income','higher_income']]=df['income'].str.split('-',n=1,expand=True)
df['higher_income']=df['higher_income'].str.rstrip('L')
df['lower_income']=df['lower_income'].str.rstrip('L')
df=df.drop('income',axis=1)
df['gender']=df['gender'].map({'Male':1,'Female':0})
df['area']=df['area'].map({'Rural':0,'Urban':1})
df['qualification']=df['qualification'].map({'Bachelor':0,'High School':1,'Others':2})
df['num_policies']=df['num_policies'].map({'More than 1':0,'1':1})
df['policy']=df['policy'].map({'A':0,'B':1,'C':2})
df['type_of_policy']=df['type_of_policy'].map({'Platinum':2,'Gold':1,'Silver':0})
df['higher_income']=df['higher_income'].map({'10':10,'5':5,'25':25})
df['lower_income']=df['lower_income'].map({'5':5,'2':2,'10':10})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
model=LinearRegression()
target=df['cltv']
features=df[['gender', 'area', 'qualification','marital_status','claim_amount','num_policies', 'type_of_policy','higher_income','lower_income']]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.150, random_state=30)
model.fit(X_train,y_train)
sol=model.predict(X_test)
r2 = r2_score(y_test,sol)
print("The r2 score is :",r2)
print("CUSTOMER VALUE PREDICTION ")
print()
gender=int(input("1 for male and 0 for female :"))
area=int(input("1 for urban and 0 for rural :"))
qualification=int(input("2 for others, 1 for high school, 0 for bachelor :"))
marital=int(input("1 for married and 0 for unmarried :"))
claim_amount=float(input("Claim amount :"))
num_policies=int(input("No fo policies : 1 for 1 and 0 for more than 1 :"))
type_policies=int(input("TYpe of policy : 2 for platinum, 1 for gold and 0 for silver :"))
higherincome=int(input("Higher income : 10 , 5 , 25 :"))
lowerincome=int(input("Lower income : 2 5 10 :"))

testing = pd.DataFrame({
    'gender': [gender],
    'area': [area],
    'qualification': [qualification],
    'marital_status': [marital],
    'claim_amount': [claim_amount],
    'num_policies': [num_policies],
    'type_of_policy': [type_policies],
    'higher_income': [higherincome],
    'lower_income': [lowerincome]
})

ans = model.predict(testing)
print("Predicted customer value : ", ans)
