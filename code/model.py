import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from text_format import TextFormat
from abc import ABCMeta, abstractmethod

class RegressionModel():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None

    def flatten_vector(self, data):
        # Ensure the data is of numpy array type.
        data = np.array(data)

        # Flatten the vector
        data = data.flatten()

        return data

    def best_fit_line(self, x_values,y_values):
        m = ( (x_values.mean()*y_values.mean() - (x_values*y_values).mean()) / (x_values.mean()**2 - (x_values**2).mean()) )

        b = y_values.mean() - m * x_values.mean()

        return m, b

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

    def get_performance_results(self, model_name, predictions, r2):
        print(f'\n{TextFormat.BLUE}{TextFormat.BOLD}{model_name} Regression Results:{TextFormat.END}')

        # Showcase the slope and intercept.
        m, c = self.get_slope_intercept(predictions)[1], self.get_slope_intercept(predictions)[2]
        print(f'The fitted straight line has equation y = {m:.1f}x {c:=+6.1f}')

        print(f'Model Slope: {self.best_fit_line(self.y_test, predictions)[0]}')
        print(f'Model Y-Intercept: {self.best_fit_line(self.y_test, predictions)[1]}')
        
        # Display R2 and RMSE
        print(f'R² = {r2:.2f}')
        print(f'MSE = {mean_squared_error(self.y_test, predictions)}')
        print(f'RMSE = {np.sqrt(mean_squared_error(self.y_test, predictions))}')
        
        # Display the model prediction performance. 
        self.display(predictions, r2, model_name)

    def display(self, predictions, r2, model_name):
        # Receive the parameters from the fit of the polynomial.
        p = self.get_slope_intercept(predictions)[0]

        # Create plot
        plt.scatter(self.y_test, predictions, c='gray', marker='o', edgecolors='k', s=18)
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
        plt.show()

    @abstractmethod
    def evaluation(self):
        pass

class LinearRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        self.model = LinearRegression().fit(self.x_train, self.y_train)
        RegressionModel.model = self.model
        self.predictions = super().flatten_vector(self.model.predict(x_test))
        self.r2 = r2_score(self.y_test, self.predictions)

    def evaluation(self):
        super().get_performance_results('Linear', self.predictions, self.r2)

class LassoRegressionModel(RegressionModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        self.alpha = self.cross_validation()
        self.model = Lasso(alpha = self.alpha).fit(self.x_train, self.y_train)
        RegressionModel.model = self.model
        self.predictions = self.model.predict(x_test)
        self.r2 = r2_score(self.y_test, self.predictions)

    def evaluation(self):
        super().get_performance_results('Lasso', self.predictions, self.r2)

    def cross_validation(self):
        lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=0)
        # Fit model
        lasso_cv.fit(self.x_train, super().flatten_vector(self.y_train)) 

        # Return the Best Alpha score.
        return lasso_cv.alpha_