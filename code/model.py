import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy import stats
from text_format import TextFormat
from abc import ABCMeta, abstractmethod

class RegressionModel():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def flatten_vector(self, data):
        # Ensure the data is of numpy array type.
        data = np.array(data)

        # Flatten the vector
        data = data.flatten()

        return data

    def get_slope_intercept(self, predictions):
        # Degree of the fitting polynomial.
        deg = 1

        # Converting following matrix into 1D numpy array.
        self.y_test = self.flatten_vector(self.y_test)
        predictions = self.flatten_vector(predictions)

        # Parameters from the fit of the polynomial.
        p = np.polyfit(self.y_test, predictions, deg)
        m = p[0]
        c = p[1]

        return p, m, c

    def get_dof(self, n, m):
        # Degrees of freedom (number of observations - number of parameters)
        dof = n - m
        return dof

    def get_t_critical(self, dof):
        # Significance level
        alpha = 0.05
        # We're using a two-sided test
        tails = 2

        # The percent-point function (aka the quantile function) of the t-distribution
        # gives you the critical t-value that must be met in order to get significance
        t_critical = stats.t.ppf(1 - (alpha / tails), dof)

        return t_critical

    def get_std_err(self, p, dof, predictions):
        # Model the data using the parameters of the fitted straight line.
        y_model = np.polyval(p, self.y_test)

        # Create the linear (1 degree polynomial) model
        model = np.poly1d(p)

        # Fit the model.
        y_model = model(self.y_test)

        # Calculate the residuals (the error in the data, according to the model).
        resid = predictions - y_model

        # Standard deviation of the error.
        std_err = np.sqrt(sum(resid**2) / dof)
        # Return the standard deviaiton of error.
        return std_err

    def get_performance_results(self, model_name, predictions, r2):
        print(f'\n{TextFormat.BLUE}{TextFormat.BOLD}{model_name} Regression Results:{TextFormat.END}')

        # Showcase the slope and intercept.
        m, c = self.get_slope_intercept(predictions)[1], self.get_slope_intercept(predictions)[2]
        print(f'The fitted straight line has equation y = {m:.1f}x {c:=+6.1f}')
        
        # Display R2 and RMSE
        print(f'R² = {r2:.2f}')
        print(f'Root Mean Squared Error: {np.sqrt(mean_squared_error(self.y_test, predictions))}')
        
        # Display the model prediction performance. 
        self.display(predictions, r2, model_name)

    def display(self, predictions, r2, model_name):
        # Receive the parameters from the fit of the polynomial.
        p = self.get_slope_intercept(predictions)[0]
        # Number of observations.
        n = predictions.size
        # Number of parameters: equal to the degree of the fitted polynomial (ie the
        # number of coefficients) plus 1 (ie the number of constants)
        m = p.size

        # Receive the degrees of freedom.
        dof = self.get_dof(n, m)
        # Receive the critical t-value.
        t_critical = self.get_t_critical(dof)
        # Receive the standard deviation of error.
        std_err = self.get_std_err(p, dof, predictions)

        # Create plot
        plt.scatter(self.y_test, predictions, c='gray', marker='o', edgecolors='k', s=18)
        xlim = plt.xlim()
        ylim = plt.ylim()

        # Line of best fit
        plt.plot(np.array(xlim), p[1] + p[0] * np.array(xlim), label=f'Line of Best Fit, R² = {r2:.2f}')

        # Fit
        x_fitted = np.linspace(xlim[0], xlim[1], 100)
        y_fitted = np.polyval(p, x_fitted)

        # Confidence interval
        ci = t_critical * std_err * np.sqrt(1 / n + (x_fitted - np.mean(self.y_test))**2 / np.sum((self.y_test - np.mean(self.y_test))**2))
        plt.fill_between(x_fitted, y_fitted + ci, y_fitted - ci, facecolor='#b9cfe7', zorder=0, label=r'95% Confidence Interval')

        # Prediction Interval
        pi = t_critical * std_err * np.sqrt(1 + 1 / n + (x_fitted - np.mean(self.y_test))**2 / np.sum((self.y_test - np.mean(self.y_test))**2))
        plt.plot(x_fitted, y_fitted - pi, '--', color='0.5', label=r'95% Prediction Limits')
        plt.plot(x_fitted, y_fitted + pi, '--', color='0.5')

        # Title and labels
        plt.title(f'{model_name} Regression')
        plt.xlabel('Y Test')
        plt.ylabel('Y Predictions')

        # Finished
        plt.legend(fontsize=8)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()

    @abstractmethod
    def evaluation(self):
        pass

class LinearRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        self.model = LinearRegression().fit(self.x_train, self.y_train)
        self.predictions = super().flatten_vector(self.model.predict(x_test))
        self.r2 = r2_score(self.y_test, self.predictions)

    def evaluation(self):
        super().get_performance_results('Linear', self.predictions, self.r2)

class LassoRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        self.model = Lasso().fit(self.x_train, self.y_train)
        self.predictions = self.model.predict(x_test)
        self.r2 = r2_score(self.y_test, self.predictions)

    def evaluation(self):
        super().get_performance_results('Lasso', self.predictions, self.r2)