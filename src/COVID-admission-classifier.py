import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv("/Users/berkayalkan/PycharmProjects/PyhtonForDataScience/Capstone/data/covid.csv")

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


df_train = pd.read_csv("/Users/berkayalkan/PycharmProjects/PyhtonForDataScience/Capstone/data/covid_train.csv")
df_train.head()

df_test = pd.read_csv("/Users/berkayalkan/PycharmProjects/PyhtonForDataScience/Capstone/data/covid_test.csv")
df_test.head()

X_train = df_train.drop(columns=['Urgency'])

y_train = df_train['Urgency']


X_test = df_test.drop(columns=['Urgency'])

y_test = df_test['Urgency']

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

model_accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy is {model_accuracy}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

X_train = df_train.drop(columns=["Urgency"])
y_train = df_train["Urgency"]


X_test = df_test.drop(columns=["Urgency"])
y_test = df_test["Urgency"]


knn_model = KNeighborsClassifier(n_neighbors=7)

knn_model.fit(X_train, y_train)

log_model =  LogisticRegression(max_iter=10000, C=0.1)

log_model.fit(X_train, y_train)


y_pred_knn = knn_model.predict(X_test)
y_pred_log = log_model.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_log = confusion_matrix(y_test, y_pred_log)

def plot_conf_matrix(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Urgency", "Urgency"], yticklabels=["No Urgency", "Urgency"])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show(block=True)

plot_conf_matrix(cm_knn, "kNN")
plot_conf_matrix(cm_log, "Logistic Regression")

def get_specificity(cm):
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

metric_scores = {
    "Accuracy": [
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_log)
    ],
    "Recall": [
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_log)
    ],
    "Specificity": [
        get_specificity(cm_knn),
        get_specificity(cm_log)
    ],
    "Precision": [
        precision_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_log)
    ],
    "F1-score": [
        f1_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_log)
    ]
}

for metric, scores in metric_scores.items():
    print(f"{metric} -> kNN: {scores[0]:.4f} | Logistic Regression: {scores[1]:.4f}")

from sklearn.metrics import roc_auc_score

roc_knn = roc_auc_score(y_test, knn_model.predict_proba(X_test)[:,1])
roc_log = roc_auc_score(y_test, log_model.predict_proba(X_test)[:,1])

print(f"ROC AUC -> kNN: {roc_knn:.4f} | Logistic Regression: {roc_log:.4f}")


X_train = df_train.drop(columns=['Urgency'])
y_train = df_train['Urgency']
X_test = df_test.drop(columns=['Urgency'])
y_test = df_test['Urgency']


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict_proba(X_test)[:, 1]


logreg = LogisticRegression(max_iter=10000, C=0.1, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict_proba(X_test)[:, 1]


def get_thresholds(y_pred_proba):

    unique_probas = np.unique(y_pred_proba)
    unique_probas_sorted = np.sort(unique_probas)[::-1]
    thresholds = np.insert(unique_probas_sorted, 0, 1.1)
    thresholds = np.append(thresholds, 0)
    return thresholds


knn_thresholds = get_thresholds(y_pred_knn)

logreg_thresholds = get_thresholds(y_pred_logreg)


def get_fpr(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba >= threshold).astype(int)
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    return FP / (FP + TN) if (FP + TN) != 0 else 0


def get_tpr(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba >= threshold).astype(int)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN) if (TP + FN) != 0 else 0


knn_fpr =  [get_fpr(y_test, y_pred_knn, t) for t in knn_thresholds]
knn_tpr = [get_tpr(y_test, y_pred_knn, t) for t in knn_thresholds]

logreg_fpr = [get_fpr(y_test, y_pred_logreg, t) for t in logreg_thresholds]
logreg_tpr = [get_tpr(y_test, y_pred_logreg, t) for t in logreg_thresholds]

knn_auc = roc_auc_score(y_test, y_pred_knn)

logreg_auc = roc_auc_score(y_test, y_pred_logreg)

# The ROC curve includes three shaded regions representing different real-world scenarios:
# - Scenario 1 (Brazil): Prefers low false positive rates, even if true positive rates are moderate.
# - Scenario 2 (Germany): Requires high true positive rates (TPR > 0.8); the system must be highly sensitive.
# - Scenario 3 (India): Allows for moderate false positive rates, but overall model performance should still be reasonable.
# These regions help visualize which model may be better suited for each scenario based on the trade-offs shown in the ROC curves.

fig, ax = plt.subplots(figsize = (14,8))
ax.plot(knn_fpr,
        knn_tpr,
        label=f'KNN (area = {knn_auc:.2f})',
        color='g',
        lw=3)

ax.plot(logreg_fpr,
        logreg_tpr,
        label=f'Logistic Regression (area = {logreg_auc:.2f})',
        color = 'purple',
        lw=3)

label_kwargs = {}
label_kwargs['bbox'] = dict(
    boxstyle='round, pad=0.3', color='lightgray', alpha=0.6
)
eps = 0.02
for i in range(0, len(logreg_fpr),15):
    threshold = str(np.round(logreg_thresholds[i], 2))
    ax.annotate(threshold, (logreg_fpr[i], logreg_tpr[i]-eps), fontsize=12, color='purple', **label_kwargs)

for i in range(0, len(knn_fpr)-1):
    threshold = str(np.round(knn_thresholds[i], 2))
    ax.annotate(threshold, (knn_fpr[i], knn_tpr[i]+eps), fontsize=12, color='green', **label_kwargs)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.fill_between([0,0.5],[0.5,0], color = 'red', alpha = 0.4, label='Scenario 1 - Brazil');
ax.axhspan(0.8, 0.9, facecolor='y', alpha=0.4, label = 'Scenario 2 - Germany');
ax.fill_between([0,1],[1,0],[0.5,-0.5], alpha = 0.4, color = 'blue', label = 'Scenario 3 - India');
ax.set_xlim([0.0, 1.0]);
ax.set_ylim([0.0, 1.05]);
ax.set_xlabel('False Positive Rate', fontsize=20)
ax.set_ylabel('True Positive Rate', fontsize=20)
ax.set_title('Receiver Operating Characteristic', fontsize=20)
ax.legend(loc="lower right", fontsize=15)
plt.show(block=True)

# Classifier selection based on real-world scenario constraints:
# - BRAZIL: Logistic Regression with a high threshold to minimize false positives.
# - GERMANY: Logistic Regression with a low threshold to ensure high true positive rates.
# - INDIA: kNN classifier with a moderate threshold for a balanced trade-off between TPR and FPR.
# These recommendations align with the shaded regions shown in the ROC curve plot.
