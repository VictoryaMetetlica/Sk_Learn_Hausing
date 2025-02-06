from Housing_2_Researching import housing
from Housing_3_NaN import housing_num
import numpy as np
from sklearn.preprocessing import OneHotEncoder


print('\n   Handling of text and categorical attributes')

housing_cat = housing.select_dtypes(include=np.object_)
print(housing_cat.head())
print('\n   Comparing each category with the help of the class OneHotEncoder. It makes one binary attribute to category: '
      'one attribute equal 1, others 0.')
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(cat_encoder.categories_)
print('\n   By default, the OneHotEncoder class returns a sparse array, but we can convert it to a dense array '
      'if needed by calling the toarray() method:')
print(housing_cat_1hot.toarray())
print('\n   Alternatively, you can set sparse_output=False when creating the OneHotEncoder')
cat_encoder = OneHotEncoder(sparse_output=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)

