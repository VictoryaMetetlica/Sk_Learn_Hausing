from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from Housing_2_Researching import housing_num


print('\n   MinMaxScaling:')
min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
print(housing_num_min_max_scaled)
print('\n   StandardScaler')
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
print(housing_num_std_scaled)


