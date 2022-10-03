# Import appropriate libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from model import LinearRegressionModel

class Preprocess():
    def __init__(self, df):
        self.df = df

    # Create a function that returns a list of outliers
    def detect_outlier(self, data):
        outliers = []
        # Finding the 1st quartile
        q1 = np.quantile(data, 0.25)
 
        # Finding the 3rd quartile
        q3 = np.quantile(data, 0.75)
 
        # Finding the iqr region
        iqr = q3-q1
 
        # Finding upper and lower whiskers
        upper_bound = q3+(1.5*iqr)
        lower_bound = q1-(1.5*iqr)
    
        defects = data[(data <= lower_bound) | (data >= upper_bound)]
    
        # Add outliers into a list
        for defect in defects:
            outliers.append(defect)
    
        # Remove any duplicates from the list
        outliers = list(dict.fromkeys(outliers))
        
        return outliers

    def replace_outliers(self):
        for col in self.df.columns:
            if col == 'y':
                pass
            else:
                outliers = self.detect_outlier(self.df[col])
                median = self.df[col].quantile(0.50)
                # print(f'The median of {col} is {median}\n')
               
                for outlier in outliers:
                    # Replace the all outliers with the variable (col) median
                    self.df[col] = np.where(self.df[col] == outlier, median, self.df[col])

    def handle_missing_values(self):
        # Determine if there are any missing values within the dataset
        for col in self.df.columns:
            missing = self.df[col].isnull().values.any()

            if missing == True:
                # Replace all the missing NaNs to interpolated values.
                self.df[col] = self.df[col].interpolate()
            else:
                pass

    def clean(self):
        self.handle_missing_values()
        self.replace_outliers()

def main():
    # Import the data and set in variable.
    df = pd.read_csv('cw1data.csv')

    data = Preprocess(df)
    data.clean()

    # Randomly order the dataset
    df = df.sample(frac=1)
    
    # Declare the label and attributes in separate variables.
    x = df.loc[:, df.columns != 'y']
    y = df.loc[:, df.columns == 'y']

    # Divide the data set into training data and testing data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.22, random_state=42)

    linear_regression = LinearRegressionModel(x_train, y_train, x_test, y_test)
    linear_regression.display()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\nSomething went wrong...")
