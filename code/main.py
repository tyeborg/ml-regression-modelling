# Import appropriate libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocessing import Preprocessing
from model import LinearRegressionModel, LassoRegressionModel

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

    # Scale features (x)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Apply and visualize Linear Regression Performance.
    linear = LinearRegressionModel(x_train, y_train, x_test, y_test)
    linear.evaluation()

    # Apply and visualize Lasso Regression Performance.
    lasso = LassoRegressionModel(x_train, y_train, x_test, y_test)
    lasso.evaluation()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\nSomething went wrong...")
