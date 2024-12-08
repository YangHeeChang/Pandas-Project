import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\diogm\Desktop\2학년2학기\Marine_Fish_Data.csv", sep=",")

grouped = df.groupby('Species_Name')['Fish_Population'].agg(['mean', 'median', 'count', 'var', 'std'])
print(grouped)
grouped2 = df.groupby('Species_Name')['Overfishing_Risk'].value_counts().unstack()
grouped2['Yes_Percentage'] = grouped2['Yes'] / (grouped2['No'] + grouped2['Yes']) * 100
print(grouped2)

fish_count = df.groupby(['Species_Name', 'Region'], as_index=False).sum()

plt.figure(figsize=(14, 8))
sns.barplot(data=fish_count, x='Species_Name', y='Fish_Population', hue='Region', palette='viridis')

plt.title('Fish Population by Species and Region', fontsize=16)
plt.xlabel('Species_Name', fontsize=14)
plt.ylabel('Fish_Population', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')
plt.tight_layout()

plt.show()


pollution_order = ['High', 'Medium', 'Low']
df['Water_Pollution_Level'] = pd.Categorical(df['Water_Pollution_Level'], categories=pollution_order, ordered=True)
pollution_fish_count = df.groupby(['Species_Name', 'Water_Pollution_Level'], as_index=False).sum()

plt.figure(figsize=(12, 6))
sns.barplot(data=pollution_fish_count, x='Species_Name', y='Fish_Population', hue='Water_Pollution_Level', palette='coolwarm')

plt.title('Fish Population by Species and Pollution Levels', fontsize=16)
plt.xlabel('Species_Name', fontsize=14)
plt.ylabel('Fish_Population', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Water_Pollution_Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')
plt.tight_layout()

plt.show()

le = LabelEncoder()
df['Water_Pollution_Level'] = le.fit_transform(df['Water_Pollution_Level'])
data = df[['Species_Name', 'Water_Pollution_Level', 'Fish_Population']]
X = data[['Water_Pollution_Level']]
y = data['Fish_Population']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'MSE: {mse}, R-squared: {r2}')
