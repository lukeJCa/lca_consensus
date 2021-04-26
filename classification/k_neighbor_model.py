import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
    'n_neighbors': (1,10, 1),
    'leaf_size': (20,40,1),
    'p': (1,2),
    'weights': ('uniform', 'distance'),
    'metric': ('minkowski', 'chebyshev'),
}

class KNeighborModel():
	def __init__(self, df, label):
		self.df = df
		self.label = label

		self.labels = df[label]
		self.dropped_df = df.drop(label, axis = 1)
		self.scaled_data = self.scale_values()
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.scaled_data, self.labels, test_size=0.2, random_state=42)


		## Implement the model

		neigh = KNeighborsClassifier()
		clf = GridSearchCV(neigh, param_grid)
		clf.fit(self.X_train, self.y_train)
		pred = clf.predict(self.X_test)
		self.score = mean_absolute_error(pred, self.y_test)
		self.accuracy = accuracy_score(pred, self.y_test)

	def scale_values(self):
		scaled_df = pd.DataFrame(minmax_scale(self.dropped_df), columns = self.dropped_df.columns)
		return scaled_df

	def get_score(self, verbose = False):

		if verbose:
			print("The K nearest neighbor model produced: "  + str(self.score) + " MAE and " + str(self.accuracy) + " accuracy percentage")

		return self.score
	def sanity_check(self):
		print(self.name + " is swimming.")