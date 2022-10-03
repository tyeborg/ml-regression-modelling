import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class RegressionModel():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def display(self):
        pass

class LinearRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)
        self.predictions = self.model.predict(x_test)

    def display(self):
        # Degree of the fitting polynomial.
        deg = 1

        # Reshaping from 2D vector to 1D vector
        self.y_test = np.reshape(self.y_test, (np.product(self.y_test.shape),))
        #self.predictions = np.reshape(self.predictions, (np.product(self.predictions.shape),))
        
        # Converting into type: numpy array
        #self.y_test = np.array(self.y_test)
        #self.predictions = np.array(self.predictions)

        self.y_test = np.array(self.y_test)
        self.y_test = self.y_test.flatten()
        
        self.predictions = np.array(self.predictions)
        self.predictions = self.predictions.flatten()

        # Parameters from the fit of the polynomial.
        p = np.polyfit(self.y_test, self.predictions, deg)
        m = p[0]
        c = p[1]

        print(f'The fitted straight line has equation y = {m:.1f}x {c:=+6.1f}')