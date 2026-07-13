import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_score, recall_score, f1_score)

pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 160)
RANDOM_STATE = 42

print("--- PART 2: LOAD & PREPROCESS ---")
df = pd.read_csv('cleaned_data.csv')
print("Loaded shape:", df.shape)

# Carry-over preprocessing
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

print("Shape after carry-over preprocessing:", df.shape)
print("Nulls remaining anywhere:", df.isnull().sum().sum())

# Define Labels and Features
y_reg = df['Fare']
y_clf = df['Survived']
X = df.drop(columns=['Fare', 'Survived'])

print("\nX columns:", X.columns.tolist())
print("\ny_clf (Survived) balance (whole dataset):")
print((y_clf.value_counts(normalize=True) * 100).round(2))

print("\n--- ENCODING ---")
# Encoding
X['Pclass'] = X['Pclass'].astype(int)
nominal_cols = ['Sex', 'Embarked']
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)

print("X shape after encoding:", X.shape)
print(X.head())

print("\n--- TRAIN/TEST SPLIT & SCALING ---")
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE
)
print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling done.")

print("\n--- REGRESSION: LINEAR VS RIDGE ---")
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_reg_train)
y_pred_reg = lin_reg.predict(X_test_scaled)

mse_lin = mean_squared_error(y_reg_test, y_pred_reg)
r2_lin = r2_score(y_reg_test, y_pred_reg)

coefs = pd.Series(lin_reg.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
print(f"Linear Regression -> MSE: {mse_lin:.2f} | R2: {r2_lin:.4f}")
print("Top 3 largest-magnitude OLS coefficients:")
print(coefs.head(3))

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)

comparison = pd.DataFrame({
    'Model': ['LinearRegression', 'Ridge(alpha=1.0)'],
    'MSE': [mse_lin, mse_ridge],
    'R2': [r2_lin, r2_ridge]
})
print("\nRegression Comparison:")
print(comparison)

print("\n--- CLASSIFICATION: LOGISTIC REGRESSION ---")
minority_pct = y_clf_train.value_counts(normalize=True).min() * 100
print(f"Minority class share in train: {minority_pct:.2f}%")
print("Below the 35% imbalance threshold?", minority_pct < 35)

logreg = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
logreg.fit(X_train_scaled, y_clf_train)

y_pred_clf = logreg.predict(X_test_scaled)
y_proba_clf = logreg.predict_proba(X_test_scaled)[:, 1]

print("\nConfusion matrix (rows=actual, cols=predicted, [0,1]):")
print(confusion_matrix(y_clf_test, y_pred_clf))

print("\nClassification report:")
print(classification_report(y_clf_test, y_pred_clf))

auc_base = roc_auc_score(y_clf_test, y_proba_clf)
print(f"AUC (C=1.0): {auc_base:.4f}")

# Plot ROC
fpr, tpr, thresholds = roc_curve(y_clf_test, y_proba_clf)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', linewidth=2, label=f'Logistic Regression (AUC = {auc_base:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve -- Logistic Regression (C=1.0)')
plt.annotate(f'AUC = {auc_base:.3f}', xy=(0.55, 0.25), fontsize=12,
             bbox=dict(boxstyle='round', fc='white', ec='darkorange'))
plt.legend(loc='lower right')
plt.tight_layout()
plt.show() # This will open a plot window in VS Code

print("\n--- THRESHOLD SENSITIVITY ---")
threshold_rows = []
for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
    preds_t = (y_proba_clf >= t).astype(int)
    p = precision_score(y_clf_test, preds_t)
    r = recall_score(y_clf_test, preds_t)
    f1 = f1_score(y_clf_test, preds_t)
    threshold_rows.append((t, p, r, f1))

threshold_table = pd.DataFrame(threshold_rows, columns=['Threshold', 'Precision', 'Recall', 'F1'])
print(threshold_table.round(4))
best_row = threshold_table.loc[threshold_table['F1'].idxmax()]
print(f"F1-maximizing threshold in the tested range: {best_row['Threshold']:.2f} (F1={best_row['F1']:.4f})")

print("\n--- REGULARIZATION: C=1.0 VS C=0.01 ---")
logreg_strong = LogisticRegression(max_iter=1000, C=0.01, random_state=RANDOM_STATE)
logreg_strong.fit(X_train_scaled, y_clf_train)

y_pred_strong = logreg_strong.predict(X_test_scaled)
y_proba_strong = logreg_strong.predict_proba(X_test_scaled)[:, 1]

p_base, r_base = precision_score(y_clf_test, y_pred_clf), recall_score(y_clf_test, y_pred_clf)
p_strong, r_strong = precision_score(y_clf_test, y_pred_strong), recall_score(y_clf_test, y_pred_strong)
auc_strong = roc_auc_score(y_clf_test, y_proba_strong)

reg_comparison = pd.DataFrame({
    'Model': ['LogReg (C=1.0)', 'LogReg (C=0.01)'],
    'Precision': [p_base, p_strong],
    'Recall': [r_base, r_strong],
    'AUC': [auc_base, auc_strong]
})
print(reg_comparison.round(4))

print("\n--- BOOTSTRAP CI FOR AUC DIFFERENCE ---")
np.random.seed(RANDOM_STATE)
y_clf_test_arr = np.array(y_clf_test)
n = len(y_clf_test_arr)
diffs = []

for i in range(500):
    idx = np.random.choice(n, size=n, replace=True)
    yb = y_clf_test_arr[idx]
    if len(np.unique(yb)) < 2:
        continue  # skip the rare resample that is all one class
    auc_b_base = roc_auc_score(yb, y_proba_clf[idx])
    auc_b_strong = roc_auc_score(yb, y_proba_strong[idx])
    diffs.append(auc_b_base - auc_b_strong)

diffs = np.array(diffs)
mean_diff = diffs.mean()
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

print(f"Bootstrap samples used: {len(diffs)} / 500")
print(f"Mean AUC diff (C=1.0 - C=0.01): {mean_diff:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print("Interval excludes zero?", ci_low > 0 or ci_high < 0)