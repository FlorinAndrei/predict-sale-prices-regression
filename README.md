# Predicting sale prices via regression

Data source:

https://www.kaggle.com/c/house-prices-advanced-regression-techniques

The main notebook is included, but it is very large. To read the code, please use the HTML version of the notebook instead:

https://florinandrei.github.io/predict-sale-prices-regression/main_notebook.html

The easiest way to run the notebook is to fork the version I have on Kaggle. Most actual compute work is disabled, you will have to re-enable the steps you want to run.

https://www.kaggle.com/code/florinandrei/sklearn-pipelines-stacking-target-encoding-pca

---

The dataset contains approx 2919 observations, split equally between train and test. There are no target values for the test data - that's the part that needs to be predicted. There are 79 features, divided almost equally between purely numeric, ordinal (categorical ordered), and nominative (categorical unordered). Some features have NaN values, a few of them have lots of NaNs. The target to be predicted is the sale price of each house in the test data.

Virtually the whole workflow is done in scikit-learn pipelines. There's a substantial amount of feature engineering, primarily required by the penalized linear regression models used here. Boosted trees models have also been used.

Other techniques used to construct the model pipelines:

- PCA (finding outliers, clustering, dimensionality reduction)
- k-means clustering (flag to the models potential clusters in the data)
- target encoding
- various feature transformations, either for single features, or in combinations

Several baseline models have been trained: XGBoost, LightGBM, CatBoost, Ridge, ElasticNet.

Optuna was used for:

- selecting the best steps (feature transformations) in each pipeline
- model tuning

The pipeline steps and the model parameters are trained together with Optuna as a single optimization loop.

Ensemble models (voting and stacking) have been trained, optimized, and tested at the end.

The best cross-validation performance was obtained from the voting regressor using XGBoost, LightGBM, CatBoost, and Ridge as base regressors.

---

Training the base predictors in an Optuna loop was very compute-intensive. The repo contains Terraform and Ansible scripts to spin up an AWS EC2 cluster to speed up training. The final models used here were trained on a total of 160 CPU cores in EC2, which took about half a day.

The code is designed to scale up the number of workers and use all CPUs available, by default.
