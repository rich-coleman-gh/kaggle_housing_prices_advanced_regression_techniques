import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# load in data
data = pd.read_csv("~/Documents/Git/kaggle_housing_prices_advanced_regression_techniques/train.csv")

# examine correlations of features against response variable
corr = data.corr()
#sns.heatmap(corr)

#sns.pairplot(data)
columns_to_examine = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']

# group features by data type
numeric_data = data.select_dtypes(include='number')
categorical_data = data.select_dtypes(include='object')
#categorical_data = categorical_data.astype(str)

# examine correlation and p value of each numerical feature against sales price
pearson_correlations = {"column" : ["correlation", "p_value"]}
for i in numeric_data.columns:
	corr, p_value = pearsonr(numeric_data[i], data['SalePrice'])
	pearson_correlations[i] = [corr, p_value]

# convert dictionary to dataframe for export
pearson_correlations = pd.DataFrame.from_dict(pearson_correlations)
pearson_correlations.to_csv("~/Documents/Git/kaggle_housing_prices_advanced_regression_techniques/pearson_correlations.csv")

# examine correlation and p value of each categorical feature against sales price
spearman_correlations = {"column" : ["correlation", "p_value"]}
for i in categorical_data.columns:
	corr, p_value = spearmanr(categorical_data[i], data['SalePrice'])
	spearman_correlations[i] = [corr, p_value]

