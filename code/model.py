import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pip install tabulate
from tabulate import tabulate

# pip install chart-studio (for plotly)
import plotly.graph_objs as go
from chart_studio import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from text_format import TextFormat
from abc import ABCMeta, abstractmethod
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV

class RegressionModel():
    # Initialize a class variable for the random state hyper parameter.
    seed = random.randrange(10000000)

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = self.flatten_vector(y_train)
        self.x_test = x_test
        self.y_test = self.flatten_vector(y_test)

        # Declare variables to be utlized within the child classes.
        self.model = None
        self.y_test_predictions = None
        self.y_train_predictions = None
        self.r2 = None
        self.test_error = None
        self.train_error = None

    @abstractmethod 
    def cross_validation(self):
        pass

    @abstractmethod
    def evaluation(self):
        pass

    def set_variables(self, model):
        # Set the model.
        self.model = model

        # Receive the test and train predictions from the model.
        self.y_test_predictions = self.model.predict(self.x_test)
        self.y_train_predictions = self.model.predict(self.x_train)

        # Receive the r2 score of the model.
        self.r2 = r2_score(self.y_test, self.y_test_predictions)

        # Calculate the model percentage error.
        self.test_error = self.get_error_percentage(self.y_test, self.y_test_predictions)
        self.train_error = self.get_error_percentage(self.y_train, self.y_train_predictions)

    def flatten_vector(self, data):
        # Ensure the data is of numpy array type.
        data = np.array(data)

        # Flatten the vector
        data = data.flatten()
        
        return data

    def get_performance_results(self, model_name):
        print(f'\n{TextFormat.BLUE}{TextFormat.BOLD}{model_name} Results:{TextFormat.END}')
        
        # Display R2, MSE, and RMSE
        print(f'R² = {self.r2:.2f}')
        print(f'MSE = {mean_squared_error(self.y_test, self.y_test_predictions)}')
        print(f'RMSE = {np.sqrt(mean_squared_error(self.y_test, self.y_test_predictions))}')

        # Display comparisons!
        self.tabular_comparisons()
        self.display(model_name)
        
    def get_error_percentage(self, data, predictions):
        data = list(data)
        predictions = list(predictions)
        error = 0

        for i in range(len(data)):
            error += (abs(predictions[i] - data[i]) / data[i])

        error_percentage = (error / len(data) * 100)
        return error_percentage
    
    def tabular_comparisons(self):
        # Transform the data back to its original state.
        actual_y_test = np.exp(self.y_test)
        actual_predictions = np.exp(self.y_test_predictions)

        # Calculate the difference between the Test Data and the Predictions.
        diff = abs(actual_y_test - actual_predictions)

        # Display the results in table form.
        actual_compare = pd.DataFrame({'Test Data': actual_y_test, 'Prediction': actual_predictions, 'Difference': diff})
        actual_compare = actual_compare.astype(float).round(2)
        print("\nTest Data vs Predicted Values:")
        print(tabulate(actual_compare.head(5), headers="keys", showindex=False, tablefmt="psql"))

    def display(self, model_name):
        # Create the trace for Actual Y Test values.
        trace0 = go.Scatter(
            y=self.y_test, 
            x=np.arange(len(self.y_test)), 
            mode='lines', 
            name='Y Test', 
            marker=dict(
                color='rgb(10, 150, 50)'
            )
        )

        # Create the trace for the Y Test Predictions.
        trace1 = go.Scatter(
            y = self.y_test_predictions,
            x = np.arange(len(self.y_test_predictions)),
            mode='lines',
            name='Predicted Y',
            line=dict(
                color='rgb(110, 50, 140)',
                dash='dot'
            )
        )

        # Add labels and titles.
        layout = go.Layout(
            title=f'{model_name} Predictions vs Actual Y Test Values',
            xaxis=dict(title='Index'),
            yaxis=dict(title='Normalized Y Values')
        )

        # Create the figure.
        figure = go.Figure(data=[trace0,trace1], layout=layout)

        if not os.path.exists("../figures"):
            os.mkdir("../figures")

        # Change the casing and spacing of the model name as an appropriate file name.
        model_name = model_name.lower().replace(" ", "-")
        figure.write_image(f'../figures/{model_name}-fig.png')

class LinearRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        # Create the Linear Regression Model.
        self.model = self.cross_validation()
        # Set the variables declared in the parent class.
        super().set_variables(self.model)

    def evaluation(self):
        # Receive the performance results of the Linear Regression Model.
        super().get_performance_results('Linear Regression')

    def cross_validation(self):
        # Creating a KFold object with 5 splits.
        folds = KFold(n_splits = 5, shuffle = True, random_state=RegressionModel.seed)

        # Specify range of hyperparameters.
        hyper_params = [{'n_features_to_select': list(range(2, 40))}]

        # Specify the model.
        linear_model = LinearRegression().fit(self.x_train, self.y_train)
        rfe = RFE(linear_model)

        # Set up GridSearchCV()
        linear_cv = GridSearchCV(estimator=rfe, param_grid=hyper_params, 
            scoring='r2', cv=folds, return_train_score=True)

        # Fit the model
        linear_cv = linear_cv.fit(self.x_train, self.y_train)

        # Return the best Linear Regression Model.
        return linear_cv

class LassoRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        # Create the Lasso Regression Model.
        self.model = self.cross_validation()
        # Set the variables declared in the parent class.
        super().set_variables(self.model)
        
    def evaluation(self):
        # Receive the performance results of the Lasso Regression Model.
        super().get_performance_results('Lasso Regression')

    def cross_validation(self):
        # Define the model.
        lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=RegressionModel.seed)
        # Fit the model.
        lasso_cv = lasso_cv.fit(self.x_train, self.y_train)

        # Return the best Lasso Regresison Model.
        return lasso_cv

class ElasticRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        # Create the Elastic Net Regression Model.
        self.model = self.cross_validation()
        # Set the variables declared in the parent class.
        super().set_variables(self.model)

    def evaluation(self):
        # Receive the performance results of the Lasso Regression Model.
        super().get_performance_results('Elastic Net Regression')

    def cross_validation(self):
        # Define model.
        elastic_cv = ElasticNetCV(cv=5, max_iter=10000, random_state=RegressionModel.seed)
        # Fit the model.
        elastic_cv = elastic_cv.fit(self.x_train, self.y_train)

        # Return the best Elastic Model.
        return elastic_cv

class KNeighborsRegressorModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        # Create the KNeighbors Regressor Model.
        self.model = self.cross_validation()
        # Set the variables declared in the parent class.
        super().set_variables(self.model)

    def evaluation(self):
        # Receive the performance results of the KNeighbors Regressor Model.
        super().get_performance_results('KNeighbors Regressor')

    def cross_validation(self):
        # Taking odd integers as K values so that
        #  majority rule can be applied easily.
        neighbors = np.arange(1, 27, 2)
        scores = []

        # Running for different K values to know 
        # which yields the max accuracy.
        for k in neighbors:
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance', p=1)
            knn.fit(self.x_train, self.y_train)
            score = cross_val_score(knn, self.x_train, self.y_train, cv=10)
            scores.append(score.mean())

        mse = [1-x for x in scores]

        # Get the best value for K.
        optimal_k = neighbors[mse.index(min(mse))]
        
        # Create the best model and fit the model.
        best_model = KNeighborsRegressor(n_neighbors=optimal_k).fit(self.x_train, self.y_train)
        return best_model

class RidgeModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        # Create the Ridge Regression Model.
        self.model = self.cross_validation()
        # Set the variables declared in the parent class.
        super().set_variables(self.model)

    def evaluation(self):
        # Receive the performance results of the Ridge Regression Model.
        super().get_performance_results('Ridge Regression')

    def cross_validation(self):
        # List of alphas to check: 100 values from 0 to 5.
        ridge_alphas = np.logspace(0, 5, 100)

        # Initiate the cross validation over alphas.
        ridge_cv = RidgeCV(alphas=ridge_alphas, scoring='r2')

        # Fit the model with the best alpha.
        ridge_cv = ridge_cv.fit(self.x_train, self.y_train)
        return ridge_cv
