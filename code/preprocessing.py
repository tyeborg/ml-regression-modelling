import numpy as np

class Preprocessing():
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

    def identify_relevant_feats(self):
        # Pick out the relevant attributes for regression modelling.
        # Relevant = Top attributes that have direct correlations with 'y'.
        correlation = self.df.corr(method='pearson')
        relevant_cols = correlation.nlargest(10, 'y').index

        # Receive a list of the top two correlated values.
        top_corr_vals = relevant_cols[:2]

        # Order the features in accordance with the head order in the df.
        relevant_copy = []

        for feat in self.df.head():
            for col in relevant_cols:
                if col == feat:
                    relevant_copy.append(col)

        return relevant_copy, top_corr_vals

    def drop_high_corr_feats(self):
        # Create the correlation matrix and select upper trigular matrix.
        cor_matrix = self.df.corr().abs()
        # Select upper trigular matrix.
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

        # Select the columns that have an absolute correlation greater than 0.95 and store into list.
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    
        print(f'Columns To Drop: {to_drop}')
        # Drop all the columns in the drop list from the dataframe.
        self.df = self.df.drop(to_drop, axis=1)

    def clean(self):
        self.handle_missing_values()
        self.replace_outliers()
        # Drop all the columns in that possess an extremely high correlation.
        self.drop_high_corr_feats()