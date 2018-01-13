import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

filename = r'C:\Users\Narsi\Desktop\DESKOP\Python\CODE\train.csv'

data = pd.read_csv(filename)
#To see the first rows of the data
print(data.head())

#To know the statistical terms of the data
print(data.describe())

#To know the structure of the data as rows and columns.
print(data.shape)

#to see the count of survivors
print(data.Survived.value_counts())

#--------------------------------------Preprocessing------------------------------------

#Taking the average of the age column to fill the missing values
avgAGE = data.Age.mean()

print(avgAGE)

#Filling the mean values of Age column to empty values
data.Age = data.Age.fillna(value=avgAGE)


#Python does not recognize empty strings which should be converted to:
# nan using numpy and then be dropped using dropna func

#Replacing the empty strings to nan
data['Embarked'].replace('',np.nan,inplace=True)
#Dropping the empty rows of Embarked column as they are required for the Data Analysis.
data.dropna(subset=['Embarked'],inplace=True)

#checking if there are any empty data still.
print(data.isnull().sum())

#Changing the categorical variable: "SEX" to numerical variable for Analyzing the Data.

data = pd.get_dummies(data,columns=["Sex"],drop_first=True)
data = data.rename(columns={"Sex_male":"Male_Gender"})

#We have changed the Categorical variable "EMBARKED" to numberical to attain efficiency in Analysis.

data = pd.get_dummies(data,columns=["Embarked"],drop_first=True)

#Changing the categorical variable: "Pclass" to numerical variable

data = pd.get_dummies(data,columns=["Pclass"],drop_first=True)
print(data.head())

#Creating a new variable called "Title" from Name in order to find inferneces
#We have split the name column to find the Designation for each person
data['Title'] = data['Name'].str.split(', ').str[1].str.split('.').str[0]
print(data['Title'].value_counts())

#Placing a Designation for each name in Title column which we can use it for visualization.

data['Title'].replace('Mme','Mrs',inplace=True)
data['Title'].replace(['Ms','Mlle'],'Miss',inplace=True)
data['Title'].replace(['Dr','Rev','Col','Major','Dona','Don','Sir','Lady','Jonkheer','Capt','the Countess'],'Others',inplace=True)
print(data['Title'].value_counts())


#Adding siblings and parch column and merging with Family column
data["Family"] = data["SibSp"]+data["Parch"]
#Adding a new column "IS_ALONE" to find the respective person is alone or not?.
data["Is_alone"]=np.where(data["Family"]>0, 0, 1)

#As Cabin contains more empty rows, we are deleting it in order to analyze efficiently.
del data["PassengerId"]
del data['Cabin']
del data["Name"]
del data["Parch"]
del data["SibSp"]
del data["Ticket"]
del data["Family"]

#------------------------------------Visualization------------------------------------------

#To plot the survivors list of male who are under the age of 10
#data[(data.Gender==1) & (data.Age <= 10)].Survived.value_counts().plot(kind = 'barh', title = 'MALE SURVIVORS UNDER THE AGE OF 10')
#plt.xlabel("Count of Survived")
#plt.ylabel("Male Gender")
#plt.show()

import seaborn as sns

#fp = sns.factorplot("Pclass","Gender","Survived",data=data,kind="bar",palette="muted",legend=True)
#plt.title("Survivors list classifying with Pclass and Gender")
#plt.xlabel("Pclass")
#plt.ylabel("Survived Count")
#plt.show()

#fp = sns.factorplot("Title","Survived",data=data,kind="bar",palette="muted",legend=True)
#plt.title("Survivors list depends on Titles'")
#plt.xlabel("Titles consolidated")
#plt.ylabel("Survived count")
#plt.show()


#fp = sns.factorplot("Title","Survived",data=data,kind="bar",palette="muted",legend=True)
#plt.show()


fp = sns.factorplot("Is_alone","Survived",data=data,kind="bar",palette="muted",legend=True)
plt.title("Is alone vs Survived counts")
plt.show()

#------------------------------------Test Data set------------------------------------------

testdata = r"C:\Users\Narsi\Desktop\DESKOP\Python\CODE\test.csv"
test = pd.read_csv(testdata)

#checking if there are any empty data still.
print(test.isnull().sum())

#Taking the average of the age column to fill the missing values
avgAGE = test.Age.mean()

#Filling the mean values of Age column to empty values
test.Age = test.Age.fillna(value=avgAGE)

#Taking the average of the fare column to fill the missing value
avgFare = test.Fare.mean()

#Filling the mean values of Fare column to empty values
test.Fare = test.Fare.fillna(value=avgFare)

#Changing the categorical variable: "SEX" to numerical variable for Analyzing the Data.

test = pd.get_dummies(test,columns=["Sex"],drop_first=True)
test = test.rename(columns={"Sex_male":"Male_Gender"})

#We have changed the Categorical variable "EMBARKED" to numberical to attain efficiency in Analysis.

test = pd.get_dummies(test,columns=["Embarked"],drop_first=True)

#Changing the categorical variable: "Pclass" to numerical variable

test = pd.get_dummies(test,columns=["Pclass"],drop_first=True)
print(test.head())

#Creating a new variable called "Title" from Name in order to find inferneces
#We have split the name column to find the Designation for each person
test['Title'] = test['Name'].str.split(', ').str[1].str.split('.').str[0]
print(test['Title'].value_counts())

#Placing a Designation for each name in Title column which we can use it for visualization.

test['Title'].replace('Mme','Mrs',inplace=True)
test['Title'].replace(['Ms','Mlle'],'Miss',inplace=True)
test['Title'].replace(['Dr','Rev','Col','Major','Dona','Don','Sir','Lady','Jonkheer','Capt','the Countess'],'Others',inplace=True)
print(test['Title'].value_counts())


#Adding siblings and parch column and merging with Family column
test["Family"] = test["SibSp"]+test["Parch"]
#Adding a new column "IS_ALONE" to find the respective person is alone or not?.
test["Is_alone"]=np.where(test["Family"]>0, 0, 1)

submission = pd.DataFrame()
submission["PassengerId"] = test["PassengerId"]

del test["PassengerId"]
del test["Ticket"]
del test["Cabin"]
del test["Name"]
del test["Parch"]
del test["SibSp"]
del test["Family"]

print(test.head())
print(data.head())

#------------------------------------Data Analytics-----------------------------------------

from sklearn.linear_model import LogisticRegression

#data_target = data["Survived"]
#data_explanatory = data["Age","Fare","Male_Gender","Embarked_Q","Embarked_S","Pclass_2","Pclass_3","Title","Is_alone"]

X_train = data.drop("Survived",axis=1)
X_train1 = X_train.drop("Title",axis=1)
Y_train = data["Survived"]
test1 = test.drop("Title",axis=1)

#Fitting the dataset into logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train1,Y_train)

print(classifier.score(X_train1,Y_train))
#Testing our test data using the model created

titanic_surviva = classifier.predict(test1)
submission["Survived"] = titanic_surviva

print(submission.head())
submission.to_csv('Submission.csv')

print(titanic_surviva)