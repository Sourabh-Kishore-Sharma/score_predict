import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

stud_df=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
stud_df.info()

#First five records
stud_df.head()

stud_df.plot(x="Hours",y="Scores",style="o")
plt.title("Hours VS Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores achieved")



#Making of the model

#Preparing the data
#Storing hours and scores in X and Y respectively in the form of numpy array.
X=stud_df.iloc[:,:-1].values
Y=stud_df.iloc[:,1].values

#Splitting the dataset into training and testing dataset.
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Training the algorithm
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
print("Training completed.")

#Plotting regression line and test data.
line=regressor.coef_*X+regressor.intercept_
plt.scatter(X, Y)
plt.plot(X, line)
plt.show()

#Making Predictions
print(X_test)
Y_predicted=regressor.predict(X_test)
print(Y_predicted)

df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_predicted})
print(df)

#Testing with own data, i.e 9.5 hours
hrs=9.5
own_pred = regressor.predict(np.array(hrs).reshape(1,-1))

print("The predicted score if the student studies for {} hrs/day is {}.".format(hrs,round(own_pred[0])))

##The predicted score if the student studies for 9.5 hrs/day is 96.0.
