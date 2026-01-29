import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data=pd.read_csv("data.csv")
print("Data loaded successfully:\n", data.head(), "\n")

X=data[["hours_studied","attendance","previous_score"]]
y=data["final_score"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model=LinearRegression()

model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("model trained succesfully")

error=mean_absolute_error(y_test,predictions)
print("mean absolute error",error)

new_student=pd.DataFrame([[6,80,70]], columns=["hours_studied","attendance","previous_score"])
predicted_score=model.predict(new_student)
print ("Predicted final score:",predicted_score[0])


