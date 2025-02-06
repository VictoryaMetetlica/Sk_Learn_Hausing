from Housing_1_Sampling import strat_train_set
import matplotlib.pyplot as plt



print("\n   It's better to make a copy of the training set: ")
housing = strat_train_set.copy()
print(housing.head())
print("\n   Shape of the data:    ", housing.shape)
print("\n   Creating a set of numerical data for further mathematical preprocessing")
housing_num = housing.iloc[:, :-1]
print(housing_num.head())
print("\n   Shape of the numeric data (without ocean_proximity attribute):    ", housing_num.shape)

print('\n   Visualization of geographic data')
    # defining the default font sizes
plt.rc('font', size=12)
plt.rc('axes', labelsize=12, titlesize=12)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing_num.plot(kind='scatter', x="longitude" , y="latitude" , alpha=0.4, figsize=(20,15))
plt.title('\n   Scatterplot of geographic data: Longitude/Latitude')
plt.show()
print('\n   Scatter plot of the median house value: radius of each circle is the population (s), color is the cost (c)')
housing_num.plot(kind='scatter', x="longitude" , y="latitude" , alpha=0.4, \
               s=housing_num["population"]/100, label="population", figsize=(20,15), \
               c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.legend()
plt.title('Scatter plot of the houses costs')
plt.show()
print('\n   As you can see from the scatter plot, the median_house_value depends on the location and the population.')
print('\n   Finding relationships: the correlation coefficient between each pair of the attributes')
corr_matrix = housing_num.corr()
print(corr_matrix)
print('\n   The correlation between each attribute and median_house_value:')
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print('Conclusions: The strongest positive correlation is between median_house_value and median_income 0,687, than'
      ' total_rooms 0,135 and housing_median_age 0,114')
print("Let's see the scatter plot for median_income and median_house_value.")
housing_num.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, figsize=(20,15))
plt.title('The scatterplot for median_income and median_house_value')
plt.show()
print("Let's extract and add the new data: rooms_per_household, bedrooms_per_room and population_per_household")
housing_num["rooms_per_household"] = housing_num["total_rooms"]/housing_num["households"]
housing_num["bedrooms_per_room"] = housing_num["total_bedrooms"]/housing_num["total_rooms"]
housing_num["population_per_household"]=housing_num["population"]/housing_num["households"]
print(housing_num.head())
print('The correlation between the new attributes')
corr_matrix = housing_num.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))
print('Conclusions:'
      '1) New bedrooms_per_room attribute better correlates with median_house_value than total_rooms/bedrooms. '
      'Houses with a lower bedroom/room ratio tend to be more expensive. '
      '2) rooms_per_household attribute is more informative than total_rooms, obviously, larger houses are more expensive.')


