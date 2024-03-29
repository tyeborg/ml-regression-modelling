import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Preprocessing():
    def __init__(self, df):
        self.df = df

    # Create a function to display a boxplot of a feature
    def boxplot(self, attribute):
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        # Changing the outlier markers
        circle = dict(markerfacecolor='slateblue', marker='o')
    
        ax.grid(None)
        ax.boxplot(x=self.df[attribute], 
                vert=False, 
                flierprops=circle, 
                patch_artist=True, 
                boxprops=dict(facecolor='lavender'), 
                medianprops = dict(color="purple",linewidth=2.5))
    
        ax.set_xlabel(attribute)

        # Create figures folder if it does not already exist.
        if not os.path.exists("../figures"):
            os.mkdir("../figures")

        if not os.path.exists("../figures/features"):
            os.mkdir("../figures/features")

        plt.savefig(f'../figures/features/{attribute}-fig.png', dpi=300, bbox_inches='tight')

    # Create a method that visualizes data.
    def visualize(self):
        for col in self.df.columns:
            self.boxplot(col)

    # Create a function that returns a list of outliers
    def detect_outlier(self, data):
        outliers = []
        # Finding the 1st quartile
        q1 = np.quantile(data, 0.25)
 
        # Finding the 3rd quartile
        q3 = np.quantile(data, 0.75)
 
        # Finding the Interquartile Range region
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
        # Detect the outliers of each column/feature.
        for col in self.df.columns:
            if col == 'y':
                pass
            else:
                outliers = self.detect_outlier(self.df[col])
                median = self.df[col].quantile(0.50)

                # Replace the outliers with the median of the respective feature.
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

    def identify_relevant_feats(self):
        # Pick out the relevant attributes for regression modelling.
        # Relevant = Top attributes that have direct correlations with 'y'.
        correlation = self.df.corr(method='pearson')
        relevant_cols = correlation.nlargest(10, 'y').index

        # Order the features in accordance with the head order in the df.
        relevant_copy = []

        for feat in self.df.head():
            for col in relevant_cols:
                if col == feat:
                    relevant_copy.append(col)

        return relevant_copy

    def drop_high_corr_feats(self, filter_threshold = 0.95):
        # Create the correlation matrix and select upper trigular matrix.
        corr_matrix = self.df.corr().abs()
        # Select upper trigular matrix.
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

        # Select the columns that have an absolute correlation greater than 0.95 and store into list.
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > filter_threshold)]
    
        print(f'Dropped Columns: {to_drop}')
        # Drop all the columns in the drop list from the dataframe.
        self.df = self.df.drop(to_drop, axis=1)

    # Identify the features to normalize.
    def feats_to_normalize(self, normalization_threshold = 0.80):
        # Declare the correlation matrix between all features.
        corr_matrix = self.df.corr()
        # Create a dataframe that maps correlations to the target variable: 'y'.
        corr_values = pd.DataFrame(corr_matrix["y"].sort_values(ascending=False))

        # Declare a list to store all feats that pass the normalization threshold.
        column_corr = []

        # Identify which columns pass the normalization threshold (0.80).
        for col in corr_values.columns:
            for idx, row in corr_values.iterrows():
                if(row[col]> normalization_threshold) and (row[col]<1):
                    if (idx not in column_corr):
                        column_corr.append(idx)
                    if (col not in column_corr):
                        column_corr.append(col)

        return column_corr

    def clean(self):
        # Account for missing values within the dataframe.
        self.handle_missing_values()
        # Replace all outliers with the median of their respective column.
        self.replace_outliers()
        # Drop all the columns in that possess an extremely high correlation.
        self.drop_high_corr_feats()