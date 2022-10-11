# Import appropriate libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocessing import Preprocessing
from model import LinearRegressionModel, LassoRegressionModel
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Import the data and set in variable.
    df = pd.read_csv('cw1data.csv')
    
    # Preprocess the data within the DataFrame.
    data = Preprocessing(df)
    data.clean()

    # Display the all correlations between features.
    data.display_corr()

    # Identify the most relevant features towards y.
    relevant_feats = data.identify_relevant_feats()[0]
    print(f'Most relevant columns: {relevant_feats}')
    
    # Normalize the top correlated values with y.
    df['y'] = np.log(df['y'])
    df['x2'] = np.log(df['x2'])

    # Declare the label and attributes in separate variables.
    x = df[[col for col in relevant_feats if col != 'y']]
    y = df.loc[:, df.columns == 'y']

    # Divide the data set into training data and testing data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=57)

    # Scale features (x)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Apply and visualize Linear Regression Performance.
    linear = LinearRegressionModel(x_train, y_train, x_test, y_test)
    linear.evaluation()

    # [1, 38, 43, 57]

    # Apply and visualize Lasso Regression Performance.
    lasso = LassoRegressionModel(x_train, y_train, x_test, y_test)
    lasso.evaluation()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\nSomething went wrong...")