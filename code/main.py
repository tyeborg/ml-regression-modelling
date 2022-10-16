# Import appropriate libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocessing import Preprocessing
from model import LinearRegressionModel, LassoRegressionModel, ElasticRegressionModel
from text_format import TextFormat
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
import random

def model_diff_plot(self, linear, lasso, y_test, p1, p2):
    plt.scatter(x=y_test, y=linear.predictions, c='navy', marker='o', edgecolors='k', s=18, alpha=0.7)
    plt.scatter(x=y_test, y=lasso.predictions, c='mediumvioletred', marker='o', edgecolors='k', s=18, alpha=0.7)
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Line of best fit
    plt.plot(np.array(xlim), p1[1] + p1[0] * np.array(xlim), c='navy', alpha=0.7, linewidth=1.5)
    # Line of best fit
    plt.plot(np.array(xlim), p2[1] + p2[0] * np.array(xlim), c='mediumvioletred', alpha=0.7, linewidth=1.5)


    classes = ['Linear Model Predictions', 'Lasso Model Predictions']
    plt.legend(labels=classes, fontsize=8)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

def main():
    # Import the data and set in variable.
    df = pd.read_csv('cw1data.csv')
    
    # Preprocess the data within the DataFrame.
    data = Preprocessing(df)
    print(f'\n{TextFormat.CYAN}{TextFormat.BOLD}Data Preparation:{TextFormat.END}')
    data.clean()

    # Display the all correlations between features.
    data.display_corr()

    # Identify the most relevant features towards y.
    relevant_feats = data.identify_relevant_feats()
    #print(f'Most relevant columns: {relevant_feats}')
    
    # Normalize the features that have above 0.8 correlation to y.
    column_corr = data.feats_to_normalize()
    print(f'Normalized features with over 80% correlation to Y: {column_corr}')
    for col in column_corr:
        df[col] = np.log(df[col])

    # Declare the label and attributes in separate variables.
    x = df[[col for col in relevant_feats if col != 'y']]
    y = df.loc[:, df.columns == 'y']

    seed = random.randrange(100000)

    # Great Random States: [1, 38, 43, 57, 80, 98, 104, 173, 184, 242]
    # Divide the data set into training data and testing data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=seed)

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

    # Apply and visualize Elastic Net Regression Performance.
    elastic = ElasticRegressionModel(x_train, y_train, x_test, y_test)
    elastic.evaluation()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\nSomething went wrong...")