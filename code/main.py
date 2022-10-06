# Import appropriate libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessing
from model import LinearRegressionModel, LassoRegressionModel
from something import LineRegress

def main():
    # Import the data and set in variable.
    df = pd.read_csv('cw1data.csv')

    # Preprocess the data within the DataFrame.
    data = Preprocessing(df)
    data.clean()

    # Randomly order the dataset
    df = df.sample(frac=1)
    
    # Declare the label and attributes in separate variables.
    x = df.loc[:, df.columns != 'y']
    y = df.loc[:, df.columns == 'y']

    # Divide the data set into training data and testing data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    # Apply and visualize Linear Regression Performance.
    #linear_regression = LinearRegressionModel(x_train, y_train, x_test, y_test)
    #linear_regression.evaluation()

    #something_lasso_regression = LassoRegressionModel(x_train, y_train, x_test, y_test)
    #something_lasso_regression.evaluation()


    #linear = LineRegress(x_train, y_train, x_test, y_test)
    #linear.evaluation()

    model_linear = LinearRegressionModel(x_train, y_train, x_test, y_test)
    model_linear.evaluation()
    
    lasso = LassoRegressionModel(x_train, y_train, x_test, y_test)
    lasso.evaluation()

    #linear = LinearRegressionModel(x_train, y_train, x_test, y_test)
    #linear.evaluation()

if __name__ == '__main__':
    #try:
        #main()
    #except Exception:
        #print("\nSomething went wrong...")
    main()
