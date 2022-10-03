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

    def clean(self):
        self.handle_missing_values()
        self.replace_outliers()