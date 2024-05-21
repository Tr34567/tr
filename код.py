import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm

df = pd.read_csv('flights_NY.csv').dropna()
df['������������� ��������'] = df['arr_delay'].apply(lambda x: 1 if x > 0 else 0)

# --- ������ ������� � ������������ ���������� ---
correlation_coefficient = df['distance'].corr(df['air_time'])
print("����������� ����������:", correlation_coefficient)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='distance', y='air_time')
plt.title('��������� ������������� ���������� �� ������� ������')
plt.xlabel('����������')
plt.ylabel('����� ������')
plt.grid(True)
slope, intercept, r_value, p_value, std_err = stats.linregress(df['distance'], df['air_time'])
x_values = np.array([df['distance'].min(), df['distance'].max()])
y_values = slope * x_values + intercept
plt.plot(x_values, y_values, color='purple', linewidth=2) # �������� ���� ����� �� ����������
plt.show()
print("������������ �������� ���������:")
print("slope:", slope)
print("intercept:", intercept)

# --- ������ ������� �������� � �������� 15 ����� ---
df_within_15_minutes = df[(df['dep_delay'] >= -15) & (df['dep_delay'] <= 15)]
plt.figure(figsize=(10, 6))
sns.histplot(df_within_15_minutes['arr_delay'], bins=30, kde=True, stat='density', color='orange') # �������� ���� ����������� �� ���������
plt.title('������������� ����������� ������������� �������� �������')
plt.xlabel('�������� �������')
m, std = norm.fit(df_within_15_minutes['arr_delay'])
xmin, xmax = plt.xlim()
xety = np.linspace(xmin, xmax, 100)
pwr = norm.pdf(xety, m, std)
plt.plot(xety, pwr, 'k', linewidth=2)
plt.legend(['���������� ������������� ($\\mu$={:.2f}, $\\sigma$={:.2f})'.format(m, std), '�����������'])
plt.grid(True)
plt.show()
print("��������� ��������� �������������:")
print("������� �������� ��������:", m)
print("����������� ����������:", std)

# --- ������ ����������� �� ������������� � ���������� ������� ---
delay_counts = df.groupby('carrier')['������������� ��������'].mean().sort_values(ascending=True)
plt.figure(figsize=(10, 6))
delay_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral', 'lightpink', 'lightsalmon', 'gold', 'lightgray', 'wheat', 'khaki', 'palegoldenrod', 'plum', 'palevioletred', 'powderblue'])
plt.title('������������� ����������� ������������� �������� �� �������������')
plt.xlabel('������������')
plt.ylabel('����������� ������������� ��������')
plt.tight_layout()
plt.show()

# --- ������ ������� ���������� �������� ---
plt.figure(figsize=(10, 6))
plt.hist(df['distance'], bins=50, color='lightgreen', edgecolor='lightpink')
plt.title('������������� ���������� ��������')
plt.xlabel('����������')
plt.grid(True)
plt.show()

# --- ������ ����������� �� ���������� ���������� ---
quantiles = df['distance'].quantile([0.25, 0.5, 0.75])
print("��������:\n", quantiles)
short_distance = quantiles.loc[0.25]
medium_distance = quantiles.loc[0.5]
long_distance = quantiles.loc[0.75]
print("\n������� �����:")
print("��������: ��", short_distance)
print("�������: ��", short_distance, "��", medium_distance)
print("�������: ��", medium_distance, "� ����")
category_labels = ['��������', '�������', '�������']
df['��������� ��������'] = pd.cut(df['distance'], bins=[0, quantiles[0.25], quantiles[0.5], df['distance'].max()], labels=category_labels[0:])
long_flights_destinations = df[df['��������� ��������'] == '�������']['dest'].unique()
print("����������� ��� ������� ���������:", long_flights_destinations)
average_delay_by_category = df.groupby('��������� ��������', observed=False)['dep_delay'].mean()
print("������� ����� �������� ������ � ����������� �� ��������� ��������:\n", average_delay_by_category)

# --- ������ ������� �������� �� ������� ---
df['month'] = pd.to_datetime(df['month'], format='%m').dt.month_name()
plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='month', y='dep_delay', errorbar=('ci', 95), color='salmon') # �������� ���� ����� �� ���������
plt.title('������� ����� �������� �� �������')
plt.xlabel('�����')
plt.ylabel('������� ����� ��������')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- ������ t-����� ---
january_data = df[df['month'] == 'January']['dep_delay']
february_data = df[df['month'] == 'February']['dep_delay']
t_statistic, p_value = stats.ttest_ind(january_data, february_data)
al = 0.05
if p_value < al:
    print("�� ������ ���������� 0.05 �������� �����������")
else:
    print("�� ������ ���������� 0.05 �������� �� �����������")
al = 0.01
if p_value < al:
    print("�� ������ ���������� 0.01 �������� �����������")
else:
    print("�� ������ ���������� 0.01 �������� �� �����������")

# ���� ��-������� ��� �������� ������������ ����������� �������������
# �������� ������ �� ��������� 
intervals = np.linspace(df_within_15_minutes['arr_delay'].min(), df_within_15_minutes['arr_delay'].max(), 10)
observed_frequencies = np.histogram(df_within_15_minutes['arr_delay'], bins=intervals)[0]
expected_frequencies = norm.cdf(intervals, loc=m, scale=std)
expected_frequencies = np.diff(expected_frequencies) * len(df_within_15_minutes)

# ��������� ���������� ��-�������
chi2_statistic = np.sum((observed_frequencies - expected_frequencies)**2 / expected_frequencies)

# ��������� p-��������
p_value = 1 - chi2.cdf(chi2_statistic, len(intervals) - 1)

print("���������� ��-�������:", chi2_statistic)
print("p-��������:", p_value)

al = 0.05
if p_value < al:
    print("�� ������ ���������� 0.05 �������� � ���������� ������������� �����������.")
else:
    print("�� ������ ���������� 0.05 �������� � ���������� ������������� �� �����������.")