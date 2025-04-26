import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv('data/energy_usage_plus.csv')

X = df[['temperature', 'humidity', 'hour', 'is_weekend', 'season', 'district_type']]
y = df[['consumption']]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['season', 'district_type'])
    ],
    remainder='passthrough' 
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = LinearRegression()
model.fit(X_train_processed, y_train)

your_time = pd.DataFrame([{
    'temperature': 15.0,
    'humidity': 34,
    'season': 'Summer',
    'hour': 15,
    'district_type': 'Residential',
    'is_weekend': 0,

}])

your_time_processed = preprocessor.transform(your_time)
predicted_consumption = model.predict(your_time_processed)
print(f"Прогнозований обсяг споживання електроенергії:{predicted_consumption[0]}кВт·год")


y_pred = model.predict(X_test_processed)

plt.scatter(y_test, y_pred)
plt.xlabel('Справжнє споживання (кВт·год)')
plt.ylabel('Прогнозоване споживання (кВт·год)')
plt.title('Справжнє vs Прогнозоване')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"Середня відносна помилка (MAPE): {mape:.2f}%")