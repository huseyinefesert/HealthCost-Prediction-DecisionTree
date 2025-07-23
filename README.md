# HealthCost-Prediction-DecisionTree
## Medical Insurance Cost Prediction:
This project aims to predict individuals' medical insurance charges based on features such as age, gender, body mass index (BMI), number of children, smoking status, and residential region.

## Dataset Used:
Source: Kaggle – Medical Cost Personal Dataset
The dataset contains a total of 1,338 observations and 7 variables.

## Steps Performed:

- Data cleaning
- Encoding of categorical variables
- Feature engineering to enhance model performance
- Data visualization

## Model Performance Results

| Model                  | RMSE        | R² Score | MAE        |
|------------------------|-------------|----------|------------|
| Decision Tree Regressor| 4314.51     | 0.873690 | 2447.47    |
| LassoCV                | 4516.04     | 0.861614 | 2819.61    |
| RidgeCV                | 4541.98     | 0.860002 | 2832.96    |
| Lasso Regression       | 4543.04     | 0.859954 | 2823.70    |
| Ridge Regression       | 4543.27     | 0.859940 | 2824.72    |
| Linear Regression      | 4543.46     | 0.859928 | 2823.82    |
| ElasticNet Regression  | 5061.89     | 0.826138 | 3605.35    |
| ElasticNetCV           |10580.78     | 0.240352 | 8078.16    |




