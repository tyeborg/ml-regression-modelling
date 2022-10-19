# Import appropriate libraries.
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from tabulate import tabulate

# Import classes.
from preprocessing import Preprocessing
from text_format import TextFormat
from model import LinearRegressionModel, LassoRegressionModel, ElasticRegressionModel, KNeighborsRegressorModel, RidgeModel

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

    # Create figures folder if it does not already exist.
    if not os.path.exists("../figures"):
        os.mkdir("../figures")

    if not os.path.exists("../figures/results"):
        os.mkdir("../figures/results")

    plt.savefig("../figures/results/heatmap.png", dpi=300, bbox_inches='tight')

def compare_model_errors(linear, lasso, elastic, knn, ridge):
    # Store the train_error of each model into a list and round the results to the nearest hundreth.
    train_error = [linear.train_error, lasso.train_error, elastic.train_error, knn.train_error, ridge.train_error]
    train_error = ['%.2f' % elem for elem in train_error]

    # Store the test_error of each model into a list and round the results to the nearest hundreth.
    test_error = [linear.test_error, lasso.test_error, elastic.test_error, knn.test_error, ridge.test_error]
    test_error = ['%.2f' % elem for elem in test_error]

    col = {'Train Error (%)': train_error, 'Test Error (%)': test_error}
    models = ['Linear', 'Lasso', 'Elastic', 'KNN', 'Ridge']

    df = pd.DataFrame(data=col, index=models)
    print("Train Error Percentage vs Test Error Percentage")
    print(tabulate(df, headers="keys", showindex=True, tablefmt="psql"))
    
    # Create figures folder if it does not already exist.
    if not os.path.exists("../figures"):
        os.mkdir("../figures")

    if not os.path.exists("../figures/results"):
        os.mkdir("../figures/results")

    # Generate a bar plot visualizing values defined above.
    n_groups = len(models)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.4

    # Designing the bars.
    train = plt.bar(index - bar_width/2, train_error, bar_width, color = '#2c7fb8', label = 'Train % Error')
    test = plt.bar(index + bar_width/2, test_error,  bar_width, color = '#7fcdbb', label = 'Test % Error')

    # Adding x tick locations
    plt.xticks(index + 0, models)

    # Adding title and labels.
    plt.xlabel('Models')
    plt.ylabel('Percentage Error')
    plt.title('Percentage Errors of Training & Testing Data')
    plt.legend()
    plt.tight_layout()

    plt.savefig("../figures/results/comparisonplot.png", dpi=300, bbox_inches='tight')

def receive_best_model(linear, lasso, elastic, knn, ridge):
    # Declare a dictionary with all the test error percentage from models.
    error_dict = {linear.test_error: 'Linear', lasso.test_error: 'Lasso', 
        elastic.test_error: 'Elastic', knn.test_error: 'KNN', 
        ridge.test_error: 'Ridge'}

    # Sort the dictionary items.
    error_list = sorted(error_dict.items())

    # Print the best model to worst model.
    for i in range(len(error_list)):
        if error_list[i] == error_list[-1]:
            print(error_list[i][1])
        else:
            print("{0} > ".format(error_list[i][1]), end = '')

    return error_list[0][1]
    
def main():
    # Import the data and set in variable.
    df = pd.read_csv('cw1data.csv')
    
    # Preprocess the data within the DataFrame.
    data = Preprocessing(df)
    # Visualize the data.
    data.visualize()
    print(f'\n{TextFormat.CYAN}{TextFormat.BOLD}Data Preparation:{TextFormat.END}')
    data.clean()

    # Display the all correlations between features.
    display_corr(data.df)

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

    # Apply and visualize KNeighbors Regressor Performance.
    knn = KNeighborsRegressorModel(x_train, y_train, x_test, y_test)
    knn.evaluation()

    # Apply and visualize Ridge Regression Performance.
    ridge = RidgeModel(x_train, y_train, x_test, y_test)
    ridge.evaluation()

    # Display the results.
    print("\n")
    compare_model_errors(linear, lasso, elastic, knn, ridge)
    result = receive_best_model(linear, lasso, elastic, knn, ridge)
    print(f'{TextFormat.BOLD}The {result} Regression Model is the most optimal.{TextFormat.END}')

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\nSomething went wrong...")