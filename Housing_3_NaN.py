from Housing_2_Researching import strat_train_set, housing
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


print('\n    NaN preprocessing the numerical data for machine learning algorithms.')
print('\n    Splitting the data into training set and labels')
housing_num = strat_train_set.drop("median_house_value", axis= 1).select_dtypes(include=[np.number])  # drop labels for training set
print("\n    The training set")
print(housing_num.head())
print("\n   Shape of the training set:    ", housing_num.shape)
housing_labels = strat_train_set['median_house_value'].copy()
print("\n   The labels: the column median_house_value")
print(housing_labels.head())
print("\n   Shape of the labels:    ",  housing_labels.shape)

print('\n   Cleaning the NaN data can be implemented in different ways:')
null_rows_idx = housing_num.isnull().any(axis=1)
print('\n   indexes of rows with NaN True/False')
print(null_rows_idx)
sample_with_NaN = housing_num.loc[null_rows_idx].head()
print('\n   rows with NaN')
print(sample_with_NaN)
print('\n   Shape of rows with NaN: ', sample_with_NaN.shape)
print('\n   1 result of the Pandas dropna function: rows with Nan data')
housing_for_dropna = housing_num.copy()
housing_dropna = housing_for_dropna.dropna(subset=["total_bedrooms"])
print(housing_dropna.loc[null_rows_idx])
print('\n   2 result of the Pandas drop the column function: columns without Nan:')
housing_for_drop = housing_num.copy()
housing_drop = housing_for_drop.drop("total_bedrooms", axis=1)
print(housing_drop.columns)
print('\n   3 setting other values instead of NaN (zero, median or other). This way looks better than previous: all NaN '
      'values from the total_bedrooms columns are median')
housing_for_median = housing_num.copy()
median = housing_for_median["total_bedrooms"].median()
nan_median = housing_for_median.fillna({"total_bedrooms": median})
print(nan_median.loc[null_rows_idx].head())
print("\n   4 The best way is sklearn.impute.SimpleImputer class, because it works for all values from the data. "
      "Creating Imputer, where all NaN are median -strategy='median'- of the relevant attribute.")

imputer = SimpleImputer(strategy='median')
imputer.fit(housing_num)

print('\n   imputer calculate a median for each attribute and save it into statistics_ :')
print(imputer.statistics_)
print('\n   Checking that this is the same as manually computing the median of each attribute:')
print(housing_num.median().values == imputer.statistics_)
print('\n   Looking to the features of the imputer fitting data')
print(imputer.feature_names_in_)
print('\n   Using the imputer for transformation train set (setting median values for all NaN training data):')
X = imputer.transform(housing_num)
print(X)
print('\n   The result of the transformation is a massive NumPy each contains transforming attributes. '
      'Moving this massive into Раndаs DataFrame: ')
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
print(housing_tr.head())
print('\    Shape of the transforming data',  housing_tr.shape)
print('\n   Looking transformed NaN data for total_bedrooms as madian=433:')
print(housing_tr.loc[null_rows_idx].head())