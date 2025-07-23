import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("insurance.csv")
print(df.head())

#**Data Preprocessing
df.drop("region", axis=1, inplace=True)  

df.drop_duplicates(inplace=True)

#**gender encoding
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

sns.pairplot(df, diag_kind='kde', hue='charges')
plt.show()

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#**Modeling
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(),
    'Ridge Regression': Ridge(),
    'ElasticNet Regression': ElasticNet(),
    'LassoCV': LassoCV(cv=5),
    'RidgeCV': RidgeCV(cv=5),
    'ElasticNetCV': ElasticNetCV(cv=5),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=5,min_samples_leaf=4,min_samples_split=2)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', model)
    ])
    
    pipeline.fit(X_train_scaled, y_train)
    y_pred = pipeline.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {
        'RMSE': rmse,
        'R2 Score': r2,
        'MAE': mae
    }
    
#**Results
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='RMSE')
print("Model Performance Results:")
print(results_df)