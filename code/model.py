import random
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tabulate import tabulate
import matplotlib.pyplot as plt
from text_format import TextFormat
from abc import ABCMeta, abstractmethod
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNet, ElasticNetCV

class RegressionModel():
    # Initialize a class variable for the random state hyper parameter.
    seed = random.randrange(10000000)

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = self.flatten_vector(y_train)
        self.x_test = x_test
        self.y_test = self.flatten_vector(y_test)

    @abstractmethod 
    def cross_validation(self):
        pass

    @abstractmethod
    def evaluation(self):
        pass

    def flatten_vector(self, data):
        # Ensure the data is of numpy array type.
        data = np.array(data)

        # Flatten the vector
        data = data.flatten()
        
        return data

    def get_slope_intercept(self, predictions):
        # Degree of the fitting polynomial.
        deg = 1

        # Parameters from the fit of the polynomial.
        p = np.polyfit(self.y_test, predictions, deg)
        m = p[0]
        c = p[1]

        return p, m, c

    def get_performance_results(self, model_name, predictions, r2):
        print(f'\n{TextFormat.BLUE}{TextFormat.BOLD}{model_name} Regression Results:{TextFormat.END}')

        # Showcase the slope and intercept.
        m, c = self.get_slope_intercept(predictions)[1], self.get_slope_intercept(predictions)[2]
        print(f'The fitted straight line has equation y = {m:.1f}x {c:=+6.1f}')
        
        # Display R2, MSE, and RMSE
        print(f'R² = {r2:.2f}')
        print(f'MSE = {mean_squared_error(self.y_test, predictions)}')
        print(f'RMSE = {np.sqrt(mean_squared_error(self.y_test, predictions))}')
        
        # Display the model prediction performance. 
        self.display(predictions, r2, model_name)

    def get_error_percentage(self, data, predictions):
        data = list(data)
        predictions = list(predictions)
        error = 0

        for i in range(len(data)):
            error += (abs(predictions[i] - data[i]) / data[i])

        error_percentage = ((error / len(data)) * 100).round(2)
        return error_percentage
        
    def make_comparisons(self, predictions):
        compare = pd.DataFrame({'Test Data': self.y_test, 'Prediction': predictions})
        print("\nTest Data vs Predicted Values:")
        print(tabulate(compare.head(5), headers="keys", showindex=False, tablefmt="psql"))

    def make_accurate_comparisons(self, predictions):
        # Transform the data back to its original state.
        actual_y_test = np.exp(self.y_test)
        actual_predictions = np.exp(predictions)

        # Calculate the difference between the Test Data and the Predictions.
        diff = abs(actual_y_test - actual_predictions)

        # Display the results in table form.
        actual_compare = pd.DataFrame({'Test Data': actual_y_test, 'Prediction': actual_predictions, 'Difference': diff})
        actual_compare = actual_compare.astype(float).round(2)
        print("\nTest Data vs Predicted Values (reverted):")
        print(tabulate(actual_compare.head(5), headers="keys", showindex=False, tablefmt="psql"))

    def display(self, predictions, r2, model_name):
        # Receive the parameters from the fit of the polynomial.
        p = self.get_slope_intercept(predictions)[0]

        # Create plot
        plt.scatter(self.y_test, predictions, c='mediumpurple', marker='o', edgecolors='k', s=18)
        xlim = plt.xlim()
        ylim = plt.ylim()

        # Line of best fit
        plt.plot(np.array(xlim), p[1] + p[0] * np.array(xlim), label=f'Line of Best Fit, R² = {r2:.2f}')

        # Title and labels
        plt.title(f'{model_name} Regression')
        plt.xlabel('Y Test')
        plt.ylabel('Y Predictions')

        # Finished
        plt.legend(fontsize=8)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.savefig(f'{model_name.lower()}plot.png')
        plt.show()

class LinearRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        # Create the Linear Regression Model.
        self.model = self.cross_validation()

        # Receive the predictions from the model.
        self.test_predictions = self.model.predict(self.x_test)
        self.train_predictions = self.model.predict(self.x_train)

        # Declare the r2 score and slope-intercept variables.
        self.r2 = r2_score(self.y_test, self.test_predictions)
        self.p = super().get_slope_intercept(self.test_predictions)[0]

        self.test_error = super().get_error_percentage(self.y_test, self.test_predictions)
        self.train_error = super().get_error_percentage(self.y_train, self.train_predictions)

    def evaluation(self):
        # Receive the performance results of the Linear Regression Model.
        super().get_performance_results('Linear', self.test_predictions, self.r2)
        # Display comparisons between the predicted y_test values and the actual y_test values.
        super().make_comparisons(self.test_predictions)
        super().make_accurate_comparisons(self.test_predictions)

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
        linear_cv.fit(self.x_train, self.y_train)

        # Return the best Linear Regression Model.
        return linear_cv

class LassoRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        # Create the Lasso Regression Model.
        self.model = self.cross_validation()

        # Receive the predictions, r2 score, and slope-intercept from the model.
        self.predictions = self.model.predict(self.x_test)
        self.r2 = r2_score(self.y_test, self.predictions)
        self.p = super().get_slope_intercept(self.predictions)[0]

    def evaluation(self):
        # Receive the performance results of the Lasso Regression Model.
        super().get_performance_results('Lasso', self.predictions, self.r2)
        # Display comparisons between the predicted y_test values and the actual y_test values.
        super().make_comparisons(self.predictions)
        super().make_accurate_comparisons(self.predictions)

    def cross_validation(self):
        # Define the model.
        lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=RegressionModel.seed)

        # Fit the model.
        lasso_cv.fit(self.x_train, self.y_train)

        # Return the best Lasso Regresison Model.
        return lasso_cv

class ElasticRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        # Create the Elastic Net Regression Model.
        self.model = self.cross_validation()

        # Receive the predictions, r2 score, and slope-intercept from the model.
        self.predictions = self.model.predict(self.x_test)
        self.r2 = r2_score(self.y_test, self.predictions)
        self.p = super().get_slope_intercept(self.predictions)[0]

    def evaluation(self):
        # Receive the performance results of the Lasso Regression Model.
        super().get_performance_results('Elastic', self.predictions, self.r2)
        # Display comparisons between the predicted y_test values and the actual y_test values.
        super().make_comparisons(self.predictions)
        super().make_accurate_comparisons(self.predictions)

    def cross_validation(self):
        # Define model.
        elastic_cv = ElasticNetCV(cv=5, max_iter=10000, random_state=RegressionModel.seed)

        # Fit the model.
        elastic_cv.fit(self.x_train, self.y_train)

        # Return the best Elastic Model.
        return elastic_cv