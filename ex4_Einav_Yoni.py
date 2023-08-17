# Introduction to Data Science with Python
# MTA - Spring 2021-2022.
# Final Home Exercise.

# First and Last Names of student: Einav Yoni


# In this exercise you should implement a classification pipeline which aim at predicting the amount of hours
# a worker will be absent from work based on the worker characteristics and the work day missed.
#


from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def load_dataset(train_csv_path):
    data = pd.read_csv(train_csv_path, sep=',')
    return data

class DataPreprocessor(object):
    """
    This class is a mandatory API. More about its structure - few lines below.

    The purpose of this class is to unify data preprocessing step between the training and the testing stages.
    This may include, but not limited to, the following transformations:
    1. Filling missing (NA / nan) values
    2. Dropping non descriptive columns
    """

    def __init__(self):
        self.transformer: Pipeline = None

    def fit(self, dataset_df):
      

        numerical_columns = ['Transportation expense', 'Height', 'Weight', 'Service time']  # There are more - what else?
        # numerical_columns = ['Reason', 'Education']
        # numerical_columns = ['Transportation expense', 'Residence Distance', 'Service time', 'Weight', 'Height',]
        categorical_columns = list(set(dataset_df.columns) - set(numerical_columns))

        # Handling Numerical Fields
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median"))
        ])

        # Handling Categorical Fields
        categorical_transformer = OneHotEncoder(drop=None, sparse=False, handle_unknown='ignore')
        cat_pipeline = Pipeline([
            ('1hot', categorical_transformer)
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("dropId", 'drop', 'ID'),
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns),
            ]
        )

        self.transformer = Pipeline(steps=[
            ("preprocessor", preprocessor)
        ])

        self.transformer.fit(dataset_df)

    def transform(self, df):

        return self.transformer.transform(df)
        # think about if you would like to add additional computed columns.


def train_model(processed_X, y):

    """
    This function gets the data after the pre-processing stage  - after running DataPreprocessor.transform on it,
    a vector of labels, and returns a trained model.

    Input:
    processed_X (ndarray or dataframe): the data after the pre-processing stage
    y: a vector of labels

    Output:
    model: an object with a "predict" method, which accepts the ***pre-processed*** data and outputs the prediction


    """
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(random_state=1)
    model.fit(processed_X, y)

    return model


