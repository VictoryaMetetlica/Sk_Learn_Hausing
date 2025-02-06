import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors

print('\n чтобы увидеть в консоли статистику для всех столбцов, необходимо расширить ширину отображения вывода')
pd.set_option('display.max_columns', None)

print('\n создание Dataframe ВВП/индекс жизни и выборка нужных данных')
print('\n      данные "Индекс лучшей жизни" c веб-сайта Организации экономического сотрудничества и развития (ОЭСР)')
oecd_bli = pd.read_csv('/home/nikolay/PycharmProjects/Sk_Learn_Housing/Data/oecd_bli_2015.csv', \
                       thousands=',')
print(oecd_bli.head(10))
print('\n      выбрать те, которые не разделяют по полу м/ж')
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
print(oecd_bli.head())
print('\n      повернуть таблицу, чтобы выводила в столбец Life satisfactions из Indicator')
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
print(oecd_bli.head())


print('\n    статистические данные по ВВП на душу населения с веб-сайта Международного валютного фонда')
gdp_per_capita = pd.read_csv('/home/nikolay/PycharmProjects/Sk_Learn_Housing/Data/gdp_per_capita.csv', \
                             thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')
print(gdp_per_capita.head(10))
gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
gdp_per_capita.set_index("Country", inplace=True)
country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                               left_index=True, right_index=True)
country_stats.sort_values(by="GDP per capita", inplace=True)
Greece_satisfaction = np.c_[country_stats.loc[['Greece'], ['Life satisfaction']]]  # для Греции
Greece_gdp = np.c_[country_stats.loc[['Greece'], ['GDP per capita']]]  # для Греции
remove_indices = [1, 3, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 33, 34, 35]
remove_indices1 = [0, 1, 6, 8, 33, 34, 35]      # missing_data
keep_indices = list(set(range(36)) - set(remove_indices))
country_stats = country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
print('   выбранные данные')
print(country_stats)

print('\n     Подготовка данных к обучению')
X = np.c_[country_stats['GDP per capita']]       # взять массив по столбцу 'GDP per capita'
y = np.c_[country_stats['Life satisfaction']]    # взять массив по столбцу 'Life satisfaction'

print('\n      ВВП на душу населения:', X, sep='\n')
print('\n     Удовлетворенность жизнью', y, sep='\n')

    # визуализировать данные с помощью скаттерплота
country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()
print('     Применим к нашим тестовым данным линейную модель')
    # Выбрать модель
model = sklearn.linear_model.LinearRegression()
    # Обучить модель
model.fit(X, y)
    # Выработать прогноз для Греции
print('\n ВВП на душу населения Греции')
X_new = Greece_gdp
print(X_new)
print('\n     прогноз удовлетворенности жизнью греками согласно обученной модели')
print (model.predict(X_new))
print('\n      удовлетворенность жизнью греков согласно данным иследований')
print (Greece_satisfaction)

print('\n      Применим к нашим тестовым данным регрессию методом k ближайших соседей для k=3')

    # Выбрать модель
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
    # Обучить модель
model.fit(X, y)
    # Выработать прогноз для Греции
print('\n ВВП на душу населения Греции')
X_new = Greece_gdp
print(X_new)
print('\n      прогноз удовлетворенности жизнью греками согласно обученной модели')
print (model.predict(X_new))
print('\n      удовлетворенность жизнью греков согласно данным иследований')
print (Greece_satisfaction)