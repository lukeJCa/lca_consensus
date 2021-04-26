import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##
##										STANDARD MODEL FORMAT
##	Should contain:
##			1) Data splitting and standardization as appropriate
##			2) Sanity check for split and standardized data
##			3) Analysis of results
##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##	

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

class RandomForestModel():
	def __init__(self, df, label):
		self.df = df
		self.label = label

		self.labels = df[label]
		self.dropped_df = df.drop(label, axis = 1)
		self.scaled_data = self.scale_values()
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.scaled_data, self.labels, test_size=0.2, random_state=42)


		## Implement the model

		rf = RandomForestClassifier(max_depth=10, random_state = 42, n_estimators=1000)
		clf = grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1)
		clf.fit(self.X_train, self.y_train)
		pred = clf.predict(self.X_test)
		self.score = mean_absolute_error(pred, self.y_test)
		self.accuracy = accuracy_score(pred, self.y_test)

	def scale_values(self):
		scaled_df = pd.DataFrame(minmax_scale(self.dropped_df), columns = self.dropped_df.columns)
		return scaled_df

	def get_score(self, verbose = False):

		if verbose:

			print("The random forest model produced: "  + str(self.score) + " MAE and " + str(self.accuracy) + " accuracy percentage")

		return self.score

	def sanity_check(self):
		print(self.name + " is swimming.")