import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


pd.set_option('display.max_columns', None)
    # loading a data
housing = pd.read_csv('/home/nikolay/PycharmProjects/Sk_Learn_Housing/Data/housing.csv')
print('\n   The data from the CSV file:')
print(housing.head())
print('\n   Information about the data')
print(housing.info())
print('''\n   Conclusions:
    1) the attribute total_bedrooms has 20433 values instead of 20640 like all, so 207 attributes are missing (NaN). 
    2) the attribute ocean_proximity has type=object, for csv it is text type.
    Let's find out what ocean_proximity categories exist and how many counties belong to each category:''',  end='\n')
print(housing["ocean_proximity"].value_counts())
print('\n   Statistical description of the data')
print(housing.describe())
print('\n   Histogram of the parameters distribution (for bins=50)')
housing.hist(bins=50, figsize=(20, 15))
plt.show()

print('\n     Generating a test sample (select 20% elements randomly) and a train sample')

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print('\n The data for the train_set')
print(train_set.head())
print('\n Shape of the train set:  ', train_set.shape)
print('\n The data for the test_set')
print(test_set.head())
print('\n Shape of the test set:  ', test_set.shape)

housing['median_income'].hist(figsize=(20, 15))
plt.show()
print("\n   Most median income values are in the $20,000 to $50,000 range (values in dataset scaled), but some fall "
      "outside of that. In the dataset, it's important to have enough samples for each stratum, otherwise the estimation "
      "of stratum importance may be biased.")
print("\n   Adding an income categorical attribute (income_cat): dividing median income on 1.5 (to limit the number of "
      "categories), rounding values ceil() (to receive discrete values) then grouping categories more than 5 into category 5")
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
print('\n    Calculating the categorical intervals for grouping the values of the sequence')
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
print(housing["income_cat"])
print(housing["income_cat"].value_counts())
print('\n    Histogram of the income categories')
housing["income_cat"].hist(figsize=(20, 15))
plt.show()

print('\n   Defining a stratified sample based on income categories:')
    # StratifiedShuffleSplit provides train/test indices to split data in train/test sets.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # split(housing, housing["income_cat"]) Generate indices to split data into training and test set.
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print("\n   Comparing train datasets from random and stratified proportions:")


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_proportions = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_proportions["Rand. %error"] = 100 * compare_proportions["Random"] / compare_proportions["Overall"] - 100
compare_proportions["Strat. %error"] = 100 * compare_proportions["Stratified"] / compare_proportions["Overall"] - 100
print(compare_proportions)
print("Conclusions: Strat. % error better than Rand. % error that's why i'll work with stratified sample")

print("\n    Removing the categorical attribute income_cat from the dataset to return the data to the original conditions.")
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
print("\n   In the next steps I will use strat_train_set to fit model and strat_test_set for predictions.")
