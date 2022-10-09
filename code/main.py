# Import appropriate libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocessing import Preprocessing
from model import LinearRegressionModel, LassoRegressionModel
import seaborn as sns
import matplotlib.pyplot as plt

def display_corr(df):
        plt.figure(figsize=(16, 6))
        heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
        # save heatmap as .png file
        # dpi - sets the resolution of the saved image in dots/inches
        # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
        plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')

def drop_high_corr_feats(df):
    pass

def main():
    # Import the data and set in variable.
    df = pd.read_csv('cw1data.csv')
    
    # Preprocess the data within the DataFrame.
    data = Preprocessing(df)
    data.clean()
    
    # Create the correlation matrix and select upper trigular matrix.
    cor_matrix = df.corr().abs()
    # Select upper trigular matrix.
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

    # Select the columns that have an absolute correlation greater than 0.95 and store into list.
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    print(f'Columns To Drop: {to_drop}')

    # Drop all the columns in the drop list from the dataframe
    df = df.drop(to_drop, axis=1)

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
    #try:
        #main()
    #except Exception:
        #print("\nSomething went wrong...")
    main()
