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

# examine feature data types
column_types = data.columns.groupby(data.dtypes)
column_types = {k.name: v for k, v in column_types.items()}

numeric_data = data.select_dtypes(include='number')
categorical_data = data.select_dtypes(include='object')


# examine correlation and p value of each feature against sales price
pearson_correlations = {}
for i in columns_to_examine:
	corr, p_value = pearsonr(data[i], data['SalePrice'])
	pearson_correlations[i] = [corr, p_value]