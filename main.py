# import libraries
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def Model_Creation():
	diabetes_dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Diabetes Prediction/diabetes.csv')
	X = diabetes_dataset.drop(columns='Outcome',axis=1)
	Y = diabetes_dataset['Outcome']
	scaler = StandardScaler()
	scaler.fit(X)
	standerdized_data = scaler.transform(X)
	X = standerdized_data
	Y = diabetes_dataset['Outcome']
	
	x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,stratify=Y,random_state=2)
	
	""" model creation """
	lg_model = LogisticRegression()
	dt_model = DecisionTreeClassifier()
	rf_model = RandomForestClassifier()
	sv_model = SVC()
	nb_model = GaussianNB()
	kn_model = KNeighborsClassifier()

	# training the model 
	print(lg_model.fit(x_train,y_train))
	print(dt_model.fit(x_train,y_train))
	print(rf_model.fit(x_train,y_train))
	print(sv_model.fit(x_train,y_train))
	print(nb_model.fit(x_train,y_train))
	print(kn_model.fit(x_train,y_train))

	print(lg_model.score(x_test,y_test))
	print(dt_model.score(x_test,y_test))
	print(rf_model.score(x_test,y_test))
	print(sv_model.score(x_test,y_test))
	print(nb_model.score(x_test,y_test))
	print(kn_model.score(x_test,y_test))

	from sklearn.model_selection import cross_val_score
	print(cross_val_score(lg_model,X,Y,cv=3))
	print(cross_val_score(dt_model,X,Y,cv=3))
	print(cross_val_score(rf_model,X,Y,cv=3))
	print(cross_val_score(sv_model,X,Y,cv=3))
	print(cross_val_score(nb_model,X,Y,cv=3))
	print(cross_val_score(kn_model,X,Y,cv=3))

	joblib.dump(lg_model,'lg_model.sav')
	joblib.dump(dt_model,'dt_model.sav')
	joblib.dump(rf_model,'rf_model.sav')
	joblib.dump(sv_model,'sv_model.sav')
	joblib.dump(nb_model,'nb_model.sav')
	joblib.dump(kn_model,'kn_model.sav')
