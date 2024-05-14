<h1 align="center">Bigmart Sales Prediction</h1>

![Intro Img](Images/sp.png)

## Background ❓

BigMart is a leading retail store chain that operates in various cities. The management at BigMart is interested in understanding the sales patterns across different stores and optimizing their inventory and marketing strategies accordingly. To achieve this goal, they have collected historical sales data for a set of products from different stores.

## Problem Statement 🚨

The retail industry, characterized by its dynamic nature and extensive product offerings, faces the challenge of optimizing sales forecasting to maximize revenue and streamline inventory management. BigMart seeks to enhance its sales prediction accuracy through the implementation of advanced machine learning techniques.

## Objective 🎯

The objective of this project is to develop robust regression models capable of accurately forecasting the sales of various products across different BigMart outlets. Leveraging historical sales data spanning multiple outlets and product categories, the aim is to construct predictive models that capture the intricate relationships between key factors influencing sales, such as product visibility, store size, location, product price, and seasonal variations.

## Solution 💡

This is a supervised learning regression problem, where the target variable is the sales figure for each product in each store. We will explore various machine learning techniques to build predictive models, evaluate their performance using appropriate metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R2 score and Adjusted R2 Score. After examining the metrics we will then select the best-performing model for deployment. The process will start with Exploratory Data Analysis (EDA) with a goal to gain better insight about our data set and eventually develop a model using machine learning methodology.

## 1. Exploratory Data Analysis 💾

I have obtained the data set and now it is time to perform an Exploratory data Analysis (EDA) to gain insight about the dataset and prepare the data for modeling purposes.

**1.1 Dataset**

***“Bigmart Sale”*** data is the historical dataset containing sales data of Bigmart across multiple chains. The dataset contains sales information of outlets established from the year 1985 to the year 2009. 

**1.2 Initial Observation**

- The dataset contains 8523 observation
- The Dataset does not contain 3878 null values and no duplicate values 
- The dataset contains a total of 12 columns out of which we have a single dependent variable labeled “Item_Outlet_Sales”
- The Item_Outlet_Sales feature contains continuous data , which represents the outlet sales of the respective item in the observation identified by the “Item_Identifier” column.

**1.3 Linear Regression Assumption**

A major goal of this project is to check how well suitable the data is to run a linear regression model examining the assumptions of linear regression during the exploratory data analysis phase. The assumptions to check are mentioned below:
1. Linear relationship
2. Multivariate Normality
3. Multicollearity
4. No Heteroscedasticity   

1.4 Univariate, Bivariate and Multivariate Analysis 
---
**1.4.1 Distribution of Features** 

The following diagrams shows the distribution of our features 

![Img_1](Images/cat_dis.png)

<p align="center"><strong>Figure 1. Countplot of Categorical Features</strong></p>

![Img_2](Images/num_dis.png)

<p align="center"><strong>Figure 2. Histogram of Numeric Features</strong></p>

![Img_3](Images/qq_plt.png)

<p align="center"><strong>Figure 3. QQ Plot of Numeric Features</strong></p>

**1.4.2 Boxplot of Features**

The following diagrams shows the Boxplot of all the features in the dataset:

![Img_4](Images/bx_plt.png)

<p align="center"><strong>Figure 4. Boxplot of Numeric Features</strong></p>

![Img_5](Images/cat_sls_bx.png)

<p align="center"><strong>Figure 5. Box Plot for Categorical Features Versus Outlet Sales</strong></p>

**1.4.3 Pairplot**

The diagram below presents a pairplot showcasing all the numerical variables in our dataset. This visualization allows us to assess the linearity of our dataset.

![Img_6](Images/p_plt.png)

<p align="center"><strong>Figure 6. Pairplot for numerical Features</strong></p>

**1.4.4 Feature Correlation Heatmap**

The following diagram shows a correlation heatmap which helps us determine the multicollinearity in our dataset. It also gives us an idea about the correlation between our dependent and independent features.

![Img_6](Images/cor.png)

<p align="center"><strong>Figure 7. Correlation Heatmap</strong></p>

**1.5 Observation**

- The dataset has 3,873 missing values
- Most of our features do not exhibit a normal distribution and our categorical features exhibit data imbalance
- None of our independent features show linear relationship with our dependent feature i.e. “Item_Outlet_Sales”
- Our dataset has acceptable level of multicollinearity
- “Item_MRP” is the feature that has the highest correlation with “Item_Outlet_Sale” 

## 2. Feature Engineering and Data Cleaning

The subsequent action are taken in the following steps:

- Checking dataset for null and duplicate values. Filling missing values in ‘Item_Weight’ with median of the feature 
- Fixing “Item_Fat_Content” as same values are recorded with different notations 
- Ordinal Encoded our ordinal features i.e. 'Outlet_Size', 'Outlet_Location_Type' and 'Item_Fat_Content'
- Capped outliers in “Item_Visibility” using Interquartile range
- Grouped all values except the top three frequent ones as “other” and one-hot encoded ‘Item_Type’ feature 
- Normalized our numeric features using function transformers

![Img_7](Images/trns_hist.png)

<p align="center"><strong>Figure 8. Feature Distribution after normalization </strong></p>

## 3. Approach

I prepare the data from the EDA for machine learning model development by creating a train and test split by 70% and 30% respectively.

**3.1 Machine Learning Approach**

In this approach, we use three machine learning algorithms:
- Linear Regression
- Pycaret
- Decision Tree Model
- Random Forest  Model
- Xgboost Model
- CatBoost
- Stacking with Light Gradient Boosting Machine (LGBM)
- Blending with LGBM

**3.1.1 Linear Regression**

The following are the model performance metrics for the Linear Regression model:

<p align="center"><strong>MAE : 1058.6969328964144</strong></p>
<p align="center"><strong>MSE : 1944194.087416401</strong></p>
<p align="center"><strong>RMSE : 1394.3436045022765</strong></p>
<p align="center"><strong>R2 Score : 0.3862755022899875</strong></p>
<p align="center"><strong>Adjusted R2 Score : 0.38290893894544553</strong></p>

**Homoscedasticity**

The following diagram plots the scatter plots of the residuals to check if the nature of the error is homoscedastic or heteroscedastic.

![Img_6](Images/hsc.png)

<p align="center"><strong>Figure 9. Feature Distribution after normalization </strong></p>

**3.1.2 Running Pycaret**

Since our dataset did not satisfy the assumptions of linear regression, I used Pycaret to test for the best fit model. As seen in the diagram below the metrics for Light Gradient Boosting Machine (LGBM) is the best. 

![Img_6](Images/py.png)

<p align="center"><strong>Figure 10. Pycaret best fit model</strong></p>

**3.1.3 Decision Tree Regressor**

The following are the model performance metrics for the Decision Tree Regressor model:

<p align="center"><strong>MAE : 1115.5620582333697</strong></p>
<p align="center"><strong>MSE : 2523099.1197276604</strong></p>
<p align="center"><strong>RMSE : 1588.4266176716067</strong></p>
<p align="center"><strong>R2 Score : 0.20353232737931726</strong></p>
<p align="center"><strong>Adjusted R2 Score : 0.19916333301496902</strong></p>

**3.1.4 Random Forest Regressor**

The following are the model performance metrics for the Random Forest Regressor model:

<p align="center"><strong>MAE : 823.4308400643403</strong></p>
<p align="center"><strong>MSE : 1376800.764089129</strong></p>
<p align="center"><strong>RMSE : 1173.371537105417</strong></p>
<p align="center"><strong>R2 Score : 0.5653847715840792</strong></p>
<p align="center"><strong>Adjusted R2 Score : 0.5630007056026425</strong></p>

**3.1.5 XGBoost**

The following are the model performance metrics for the XGBoost model:

<p align="center"><strong>MAE : 858.8957547025992</strong></p>
<p align="center"><strong>MSE : 1398689.8415097597</strong></p>
<p align="center"><strong>RMSE : 1182.6621840194941</strong></p>
<p align="center"><strong>R2 Score : 0.5584750380691684</strong></p>
<p align="center"><strong>Adjusted R2 Score : 0.5560530689965912</strong></p>

**3.1.6 CatBoost**

The following are the model performance metrics for the CatBoost model:
 
<p align="center"><strong>MAE : 821.1539247580611</strong></p>
<p align="center"><strong>MSE : 1366253.9853378544</strong></p>
<p align="center"><strong>RMSE : 1168.8686775416024</strong></p>
<p align="center"><strong>R2 Score : 0.5687140773017949</strong></p>
<p align="center"><strong>Adjusted R2 Score : 0.5663482741054251</strong></p>

**3.1.7 Stacking with Light Gradient Boosting Machine (LGBM)**

The following are the model performance metrics for the Meta-model trained using Light Gradient Boosting Machine (LGBM):

<p align="center"><strong>MAE : 828.5726722873355</strong></p>
<p align="center"><strong>MSE : 1405903.3064548166</strong></p>
<p align="center"><strong>RMAE : 28.784938288753295</strong></p>
<p align="center"><strong>R2 Score : 0.5561979608067656</strong></p>
<p align="center"><strong>Adjusted R2 Score : 0.5537635009099294</strong></p>

**3.1.8 Blending with Light Gradient Boosting Machine (LGBM)**

The blending method uses CatBoost, XGBoost and Random Forest Regressor as the base model used to train the meta model. The meta model is trained using Light Gradient Boosting Machine and the model performance metrics for the Meta-model is shown below:

<p align="center"><strong>MAE : 875.1956184220188</strong></p>
<p align="center"><strong>MSE : 1526182.814465849</strong></p>
<p align="center"><strong>RMSE : 1235.3877182754607</strong></p>
<p align="center"><strong>R2 Score : 0.5182292821050554</strong></p>
<p align="center"><strong>Adjusted R2 Score : 0.515586546406235</strong></p>


