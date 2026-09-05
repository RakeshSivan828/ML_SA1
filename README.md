# ML-SA---1-fuel-consumption-co2-linear-regression


## AIM:

To write a program to implement Linear Regression for analyzing and predicting CO2 emissions based on vehicle parameters such as Cylinders, Engine Size, and Combined Fuel Consumption.

---

## EQUIPMENTS REQUIRED:

1. Hardware – PC
2. Anaconda – Python 3.7 Installation / Jupyter Notebook / Google Colab

---

## ALGORITHM:

1. Import the required libraries and read the `FuelConsumption.csv` dataset using Pandas.

2. Display the dataset and identify the required columns for analysis.

3. Create scatter plots to compare Cylinders, Engine Size, and Combined Fuel Consumption with CO2 Emissions.

4. Split the dataset into training and testing data and apply Linear Regression using Cylinders as the independent variable and CO2 Emissions as the dependent variable.

5. Train another Linear Regression model using Combined Fuel Consumption as the independent variable and CO2 Emissions as the dependent variable.

6. Calculate the R² score for both models and compare their accuracy using different train-test ratios.

---

## PROGRAM:

```python
# Program to implement Linear Regression for analyzing and predicting CO2 Emissions.
# Developed by: NITHISHWAR P
# Register Number: 212224060178

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("FuelConsumption.csv")

print(df.head())
print(df.columns)

# Q1
plt.figure(figsize=(8, 5))

plt.scatter(
    df['CYLINDERS'],
    df['CO2EMISSIONS'],
    color='green'
)

plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders vs CO2 Emissions')
plt.grid(True)
plt.show()

# Q2
plt.figure(figsize=(8, 5))

plt.scatter(
    df['CYLINDERS'],
    df['CO2EMISSIONS'],
    color='green',
    label='Cylinders'
)

plt.scatter(
    df['ENGINESIZE'],
    df['CO2EMISSIONS'],
    color='blue',
    label='Engine Size'
)

plt.xlabel('Cylinders / Engine Size')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders and Engine Size vs CO2 Emissions')
plt.legend()
plt.grid(True)
plt.show()

# Q3
plt.figure(figsize=(9, 6))

plt.scatter(
    df['CYLINDERS'],
    df['CO2EMISSIONS'],
    color='green',
    label='Cylinders'
)

plt.scatter(
    df['ENGINESIZE'],
    df['CO2EMISSIONS'],
    color='blue',
    label='Engine Size'
)

plt.scatter(
    df['FUELCONSUMPTION_COMB'],
    df['CO2EMISSIONS'],
    color='red',
    label='Fuel Consumption'
)

plt.xlabel('Independent Variables')
plt.ylabel('CO2 Emissions')
plt.title('Comparison of Variables vs CO2 Emissions')
plt.legend()
plt.grid(True)
plt.show()

# Q4
X = df[['CYLINDERS']]
y = df['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_cylinder = LinearRegression()

model_cylinder.fit(X_train, y_train)

y_pred = model_cylinder.predict(X_test)

accuracy_cylinder = r2_score(y_test, y_pred)

print("\nQ4: Cylinder vs CO2 Emissions")
print("Coefficient:", model_cylinder.coef_[0])
print("Intercept:", model_cylinder.intercept_)
print("Accuracy (R2):", accuracy_cylinder)

# Q5
X = df[['FUELCONSUMPTION_COMB']]
y = df['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_fuel = LinearRegression()

model_fuel.fit(X_train, y_train)

y_pred = model_fuel.predict(X_test)

accuracy_fuel = r2_score(y_test, y_pred)

print("\nQ5: Fuel Consumption vs CO2 Emissions")
print("Coefficient:", model_fuel.coef_[0])
print("Intercept:", model_fuel.intercept_)
print("Accuracy (R2):", accuracy_fuel)

# Q6
ratios = [0.1, 0.2, 0.3, 0.4]

cylinder_accuracy = []
fuel_accuracy = []

for ratio in ratios:

    X = df[['CYLINDERS']]
    y = df['CO2EMISSIONS']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=ratio,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cylinder_accuracy.append(
        r2_score(y_test, y_pred)
    )

    X = df[['FUELCONSUMPTION_COMB']]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=ratio,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    fuel_accuracy.append(
        r2_score(y_test, y_pred)
    )

results = pd.DataFrame({
    'Test Size': ratios,
    'Train Size': [1 - r for r in ratios],
    'Cylinder Accuracy': cylinder_accuracy,
    'Fuel Consumption Accuracy': fuel_accuracy
})

print("\nQ6: Accuracy for Different Train-Test Ratios")
print(results)

results['Cylinder Accuracy (%)'] = (
    results['Cylinder Accuracy'] * 100
)

results['Fuel Consumption Accuracy (%)'] = (
    results['Fuel Consumption Accuracy'] * 100
)

print("\nAccuracy in Percentage")

print(
    results[
        [
            'Test Size',
            'Train Size',
            'Cylinder Accuracy (%)',
            'Fuel Consumption Accuracy (%)'
        ]
    ]
)

plt.figure(figsize=(9, 5))

x = np.arange(len(ratios))
width = 0.35

plt.bar(
    x - width / 2,
    results['Cylinder Accuracy (%)'],
    width,
    label='Cylinder'
)

plt.bar(
    x + width / 2,
    results['Fuel Consumption Accuracy (%)'],
    width,
    label='Fuel Consumption'
)

plt.xlabel('Train-Test Ratio')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')

plt.xticks(
    x,
    ['90-10', '80-20', '70-30', '60-40']
)

plt.legend()
plt.grid(axis='y')

plt.show()

```

## OUTPUT

### Dataset

The program displays the first five records and the available columns
from the **`FuelConsumption.csv`** dataset.

<img width="1127" height="740" alt="image" src="https://github.com/user-attachments/assets/10eb7419-c147-4625-bea9-03c5c5a0a3f2" />


### Q1: Cylinders vs CO2 Emissions

<img width="1082" height="681" alt="image" src="https://github.com/user-attachments/assets/d4b9c5a7-9082-4bd9-b032-57b5ef59e133" />

A scatter plot is generated to show the relationship between the number
of cylinders and CO2 emissions.

### Q2: Cylinders and Engine Size vs CO2 Emissions

<img width="1110" height="676" alt="image" src="https://github.com/user-attachments/assets/30805981-bd50-41db-a6de-91fcc6504908" />


A scatter plot is generated to compare Cylinders and Engine Size with
CO2 emissions.

### Q3: Comparison of Variables vs CO2 Emissions

<img width="1083" height="685" alt="image" src="https://github.com/user-attachments/assets/0251b1fd-820e-4e4b-bab2-0719e11d9870" />


The program compares:

- Cylinders
- Engine Size
- Fuel Consumption

with CO2 emissions.

### Q4: Cylinder vs CO2 Emissions

```text
Coefficient: 29.47839878969531
Intercept: 86.08850036109291
Accuracy (R2): 0.731740029783895
```
### Q5: Fuel Consumption vs CO2 Emissions

```text
Coefficient: 16.180900781199195
Intercept: 69.10302617984444
Accuracy (R2): 0.8074147862474242
```

### Q6: Accuracy for Different Train-Test Ratios

The model accuracy is calculated using the following train-test ratios:

- **90% Training - 10% Testing**
- **80% Training - 20% Testing**
- **70% Training - 30% Testing**
- **60% Training - 40% Testing**

The results are displayed in a table and comparison graph.

## RESULT

Thus, the **Linear Regression algorithm** was successfully implemented to analyze and predict **CO₂ emissions** using vehicle parameters.

The model using **Combined Fuel Consumption** achieved a higher **R² score** than the model using **Cylinders**, indicating better prediction performance.
