import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Capstone/data/covid.csv")

print(df.head())
print(df.info())

df.isnull().sum()

num_null = df.isnull().any(axis=1).sum()
print("Number of rows with null values:", num_null)

urgency = df['Urgency']
features = df.drop(columns=['Urgency'])

imputer = KNNImputer(n_neighbors=5)
features_imputed = imputer.fit_transform(features)

features_df = pd.DataFrame(features_imputed, columns=features.columns)

df = features_df.copy()
df['Urgency'] = urgency.reset_index(drop=True)

# Plot 1: Number of urgent hospital admissions by age group

df['age_group'] = pd.cut(df['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                         labels=['0-10','11-20', '21-30', '31-40', '41-50',
                          '51-60', '61,70', '71-80','81-90', '91,100'])

urgent_cases = df[df['Urgency'] == 1]

plt.figure(figsize=(10,6))
sns.countplot(data=urgent_cases, x='age_group', order=['0-10','11-20', '21-30', '31-40', '41-50',
                          '51-60', '61,70', '71-80','81-90', '91,100'], palette='viridis')
plt.title('High Urgency Hospital Admission by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Patients')
plt.grid(axis='y')
plt.show(block=True)

df.drop(columns=['age_group'], inplace=True)


# Plot 2: Most common symptoms among urgent hospitalizations

urgent_patients = df[df['Urgency'] == 1]

symptoms = ['cough', 'fever', 'chills', 'sore_throat', 'headache', 'fatigue']

symptom_counts = urgent_patients[symptoms].sum()

plt.figure(figsize=(8,6))
sns.barplot(x=symptom_counts.index, y=symptom_counts.values, palette="magma")
plt.title('Most Common Symptoms Among Urgent Hospitalizations')
plt.xlabel('Symptom')
plt.ylabel('Number of Patients')
plt.grid(axis='y')
plt.show(block=True)


# Plot 3: Proportion of patients with cough by urgency level

cough_rates = df.groupby('Urgency')['cough'].mean().reset_index()
cough_rates['Urgency'] = cough_rates['Urgency'].map({0: 'No Urgency', 1: 'High Urgency'})

plt.figure(figsize=(8,6))
sns.barplot(data=cough_rates, x='Urgency', y='cough', palette='viridis')

plt.ylabel('Proportion of Patients with Cough')
plt.title('Cough Proportion by Urgency Level')
plt.ylim(0,1)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show(block=True)

df_train, df_test = train_test_split(df, test_size=0.3, random_state=60)

df_train.to_csv("covid_train.csv", index=False)

df_test.to_csv("covid_test.csv", index=False)

