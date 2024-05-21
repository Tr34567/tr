import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm

df = pd.read_csv('flights_NY.csv').dropna()
df['Положительная задержка'] = df['arr_delay'].apply(lambda x: 1 if x > 0 else 0)

# --- Секция расчета и визуализации корреляции ---
correlation_coefficient = df['distance'].corr(df['air_time'])
print("Коэффициент корреляции:", correlation_coefficient)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='distance', y='air_time')
plt.title('Диаграмма распределения расстояния от времени полета')
plt.xlabel('Расстояние')
plt.ylabel('Время полета')
plt.grid(True)
slope, intercept, r_value, p_value, std_err = stats.linregress(df['distance'], df['air_time'])
x_values = np.array([df['distance'].min(), df['distance'].max()])
y_values = slope * x_values + intercept
plt.plot(x_values, y_values, color='purple', linewidth=2) # Изменили цвет линии на фиолетовый
plt.show()
print("Коэффициенты линейной регрессии:")
print("slope:", slope)
print("intercept:", intercept)

# --- Секция анализа задержек в пределах 15 минут ---
df_within_15_minutes = df[(df['dep_delay'] >= -15) & (df['dep_delay'] <= 15)]
plt.figure(figsize=(10, 6))
sns.histplot(df_within_15_minutes['arr_delay'], bins=30, kde=True, stat='density', color='orange') # Изменили цвет гистограммы на оранжевый
plt.title('Нормированная гистограмма распределения задержки прилета')
plt.xlabel('Задержка прилета')
m, std = norm.fit(df_within_15_minutes['arr_delay'])
xmin, xmax = plt.xlim()
xety = np.linspace(xmin, xmax, 100)
pwr = norm.pdf(xety, m, std)
plt.plot(xety, pwr, 'k', linewidth=2)
plt.legend(['Нормальное распределение ($\\mu$={:.2f}, $\\sigma$={:.2f})'.format(m, std), 'Гистограмма'])
plt.grid(True)
plt.show()
print("Оцененные параметры распределения:")
print("Среднее значение задержки:", m)
print("Стандартное отклонение:", std)

# --- Секция группировки по авиакомпаниям и построения графика ---
delay_counts = df.groupby('carrier')['Положительная задержка'].mean().sort_values(ascending=True)
plt.figure(figsize=(10, 6))
delay_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral', 'lightpink', 'lightsalmon', 'gold', 'lightgray', 'wheat', 'khaki', 'palegoldenrod', 'plum', 'palevioletred', 'powderblue'])
plt.title('Распределение вероятности положительной задержки по авиакомпаниям')
plt.xlabel('Авиакомпания')
plt.ylabel('Вероятность положительной задержки')
plt.tight_layout()
plt.show()

# --- Секция анализа расстояния перелета ---
plt.figure(figsize=(10, 6))
plt.hist(df['distance'], bins=50, color='lightgreen', edgecolor='lightpink')
plt.title('Распределение расстояния перелета')
plt.xlabel('Расстояние')
plt.grid(True)
plt.show()

# --- Секция группировки по категориям расстояния ---
quantiles = df['distance'].quantile([0.25, 0.5, 0.75])
print("Квантили:\n", quantiles)
short_distance = quantiles.loc[0.25]
medium_distance = quantiles.loc[0.5]
long_distance = quantiles.loc[0.75]
print("\nГраницы групп:")
print("Короткие: до", short_distance)
print("Средние: от", short_distance, "до", medium_distance)
print("Длинные: от", medium_distance, "и выше")
category_labels = ['Короткий', 'Средний', 'Длинный']
df['Категория перелета'] = pd.cut(df['distance'], bins=[0, quantiles[0.25], quantiles[0.5], df['distance'].max()], labels=category_labels[0:])
long_flights_destinations = df[df['Категория перелета'] == 'Длинный']['dest'].unique()
print("Направления для длинных перелетов:", long_flights_destinations)
average_delay_by_category = df.groupby('Категория перелета', observed=False)['dep_delay'].mean()
print("Среднее время задержки вылета в зависимости от категории перелета:\n", average_delay_by_category)

# --- Секция анализа задержки по месяцам ---
df['month'] = pd.to_datetime(df['month'], format='%m').dt.month_name()
plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='month', y='dep_delay', errorbar=('ci', 95), color='salmon') # Изменили цвет точек на лососевый
plt.title('Среднее время задержки по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Среднее время задержки')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Секция t-теста ---
january_data = df[df['month'] == 'January']['dep_delay']
february_data = df[df['month'] == 'February']['dep_delay']
t_statistic, p_value = stats.ttest_ind(january_data, february_data)
al = 0.05
if p_value < al:
    print("на уровне значимости 0.05 гипотеза отвергается")
else:
    print("на уровне значимости 0.05 гипотеза не отвергается")
al = 0.01
if p_value < al:
    print("на уровне значимости 0.01 гипотеза отвергается")
else:
    print("на уровне значимости 0.01 гипотеза не отвергается")

# Тест хи-квадрат для проверки соответствия нормальному распределению
# Разделим данные на интервалы 
intervals = np.linspace(df_within_15_minutes['arr_delay'].min(), df_within_15_minutes['arr_delay'].max(), 10)
observed_frequencies = np.histogram(df_within_15_minutes['arr_delay'], bins=intervals)[0]
expected_frequencies = norm.cdf(intervals, loc=m, scale=std)
expected_frequencies = np.diff(expected_frequencies) * len(df_within_15_minutes)

# Вычисляем статистику хи-квадрат
chi2_statistic = np.sum((observed_frequencies - expected_frequencies)**2 / expected_frequencies)

# Вычисляем p-значение
p_value = 1 - chi2.cdf(chi2_statistic, len(intervals) - 1)

print("Статистика хи-квадрат:", chi2_statistic)
print("p-значение:", p_value)

al = 0.05
if p_value < al:
    print("На уровне значимости 0.05 гипотеза о нормальном распределении отвергается.")
else:
    print("На уровне значимости 0.05 гипотеза о нормальном распределении не отвергается.")
