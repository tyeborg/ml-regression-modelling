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

def main():
    # Import the data and set in variable.
    df = pd.read_csv('cw1data.csv')
    
    # Preprocess the data within the DataFrame.
    data = Preprocessing(df)
    data.clean()

    relevant_feats = data.identify_relevant_feats()[0]
    print(f'Most relevant columns: {relevant_feats}')

    # Normalize the top correlated values with y.
    df['y'] = np.log(df['y'])
    df['x2'] = np.log(df['x2'])

    # Declare the label and attributes in separate variables.
    x = df[[col for col in relevant_feats if col != 'y']]
    y = df.loc[:, df.columns == 'y']

    # Divide the data set into training data and testing data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=21)

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