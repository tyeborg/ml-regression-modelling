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
from tabulate import tabulate

def display_corr(df):
    # Plot a "pretty" version of the correlation matrix 
    # Based on: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    # Given that the correlation table is symmetrical, we remove one side 

    # Declare correlation matrix
    corr_matrix = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 8))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, center=0, 
        square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

def compare_model_errors(linear, lasso, elastic):
    train_error = [linear.train_error, lasso.train_error, elastic.train_error]
    test_error = [linear.test_error, lasso.test_error, elastic.test_error]

    col = {'Train Error': train_error, 'Test Error': test_error}
    models = ['Linear', 'Lasso', 'Elastic']

    df = pd.DataFrame(data=col, index=models)
    print(tabulate(df, headers="keys", showindex=True, tablefmt="psql"))

    df.plot(kind='bar')
    plt.savefig("comparisonplot.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Import the data and set in variable.
    df = pd.read_csv('cw1data.csv')
    
    # Preprocess the data within the DataFrame.
    data = Preprocessing(df)
    print(f'\n{TextFormat.CYAN}{TextFormat.BOLD}Data Preparation:{TextFormat.END}')
    data.clean()

    # Display the all correlations between features.
    #display_corr(data.df)

    # Identify the most relevant features towards y.
    relevant_feats = data.identify_relevant_feats()
    
    # Normalize the features that have above 0.8 correlation to y.
    column_corr = data.feats_to_normalize()
    print(f'Normalized features with over 80% correlation to Y: {column_corr}')
    for col in column_corr:
        df[col] = np.log(df[col])

    # Declare the label and attributes in separate variables.
    x = df[[col for col in relevant_feats if col != 'y']]
    y = df.loc[:, df.columns == 'y']

    # Initialize a random integer for the random state hyper parameter.
    seed = random.randrange(10000000)
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

    compare_model_errors(linear, lasso, elastic)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\nSomething went wrong...")