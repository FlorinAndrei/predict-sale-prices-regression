import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
from IPython.display import display

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline


def plot_variance(pca, width=12, dpi=100):
    n = pca.n_components_
    if n > 20:
        n_ticks = 20
    else:
        n_ticks = n
    grid = np.arange(1, n + 1)
    grid_ticks = np.arange(1, n + 1, n // n_ticks)
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cv = np.cumsum(evr)
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(grid, ev, 'o-')
    axs[0].axhline(y=1, color='C3')
    axs[0].set_xticks(grid_ticks)
    axs[0].set(xlabel="Component", title="Explained Variance", ylim=(0.0, None))
    axs[1].plot(grid, cv, "o-")
    axs[1].axhline(y=0.7, color='C3')
    axs[1].set_xticks(grid_ticks)
    axs[1].set(xlabel="Component", title="Cumulative Variance", ylim=(0.0, 1.0))
    axs[2].bar(grid, evr)
    axs[2].set(xlabel="Component", title="Relative Explained Variance", ylim=(0.0, 1.0))
    fig.set(figwidth=width, dpi=100)
    return axs


def make_mi_scores(X, y):
    X = X.copy()
    y = y.copy()
    cat_cols = X.select_dtypes(["category"]).columns.to_list()
    for colname in cat_cols:
        X[colname], _ = X[colname].factorize()
    discrete_features = [True if c in cat_cols else False for c in X.columns]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.barh(width, scores)
    ax.set_yticks(width, ticks)
    ax.set_title("Mutual Information Scores")
    fig.show()


def X_to_numeric(self, X):
    """
    Convert X to purely numeric dtypes.
    """
    X = X.copy(deep=True)
    for c in X.select_dtypes(include='category'):
        X[c] = X[c].cat.codes
    return X


def X_restore_categories(self, X, cat_cols, X_dtypes):
    """
    Restore categories to the state before the transform.
    (As close as possible.)
    X_dtypes provides the original categories.
    Only cat_cols are changed.
    No changes are made to the actual values / codes in the columns,
    besides translating them from float to categories.
    -1 is marked with the NA category.
    """
    X = X.copy(deep=True)
    for c in cat_cols:
        # get list of original categories
        cat_list_original = X_dtypes[c].categories.to_list()
        # category codes must be integer
        # also stick np.nan back in, if any
        X[c] = X[c].round().replace(-1.0, np.nan).astype('category')
        # get list of current categories
        cat_list_new = X[c].cat.categories.to_list()
        # rename new categories, make them same as old
        cat_dict = {k: cat_list_original[round(k)] for k in cat_list_new if k != -1.0}
        X[c] = X[c].cat.rename_categories(cat_dict)
        cat_list_new_renamed = X[c].cat.categories.to_list()
        # add original categories missing from new
        X[c] = X[c].cat.add_categories([cat for cat in cat_list_original if cat not in cat_list_new_renamed])
        # match order of new categories to old
        X[c] = X[c].cat.reorder_categories(new_categories=cat_list_original, ordered=X_dtypes[c].ordered)
    return X


def pd_get_dummies(X):
    X = X.copy()
    # display(X.dtypes.to_dict())
    return pd.get_dummies(X, drop_first=True, dtype=int)


def trivial_impute(df_orig):
    df = df_orig.copy(deep=True)
    for name in df.select_dtypes("number"):
        if df[name].isna().sum() > 0:
            df[name] = df[name].fillna(0)
    for name in df.select_dtypes("category"):
        if df[name].isna().sum() > 0:
            df[name] = df[name].fillna("NA")
    return df


class SimpleImputerKeepCategories(SimpleImputer):
    """
    Extend SimpleImputer() to keep original categories unchanged.
    Assumes set_output(transform='pandas').
    """

    def fit(self, X, y=None):
        self.X_dtypes = X.dtypes.to_dict()
        return super().fit(X, y)

    def transform(self, X):
        X_imputed = super().transform(X)
        for c in X_imputed.columns.to_list():
            X_imputed[c] = X_imputed[c].astype(self.X_dtypes[c])
        return X_imputed

    def fit_transform(self, X, y=None, **fit_params):
        self.X_dtypes = X.dtypes.to_dict()
        X_new = super().fit_transform(X, y=None, **fit_params)
        for c in X_new.columns.to_list():
            X_new[c] = X_new[c].astype(self.X_dtypes[c])
        return X_new


class SKPipeDataViewer(BaseEstimator, TransformerMixin):
    """
    Print out the X dataframe within the pipeline.
    """

    def __init__(self, show_dtypes=False, show_na=False, **kwargs):
        super().__init__(**kwargs)
        self.show_dtypes = show_dtypes
        self.show_na = show_na
        for k, v in kwargs.items():
            setattr(self, k, v)

    def transform(self, X):
        print()
        # in case X is not a dataframe, wrap it
        display(pd.DataFrame(X).head(10))
        if self.show_na == True:
            print(X.isna().sum().sort_values(ascending=False).head())
        if self.show_dtypes == True:
            self.dtypes = X.dtypes.to_dict()
            pprint(self.dtypes, indent=2)
        return X

    def fit(self, X, y=None, **kwargs):
        return self

    def set_output(self, transform):
        pass


class SimpleImputerPandas(BaseEstimator, TransformerMixin):
    """
    Maintain consistent before/after Pandas data types for categorical features.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._estimator = SimpleImputer(**kwargs)
        _ = self._estimator.set_output(transform='pandas')

    def fit(self, X, y=None):
        X = X.copy()
        self.X_dtypes = X.dtypes.to_dict()
        self.cat_cols = [c for c in X.select_dtypes(include='category')]
        self.num_cols = [c for c in X.select_dtypes(include='number')]
        X = self.X_to_numeric(X)
        X[self.cat_cols] = X[self.cat_cols].replace({-1: np.nan})
        self._estimator.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X = self.X_to_numeric(X)
        X[self.cat_cols] = X[self.cat_cols].replace({-1: np.nan})
        X_trans = self._estimator.transform(X=X)
        return self.X_restore_categories(X_trans, self.cat_cols, self.X_dtypes)

    def set_output(self, transform):
        pass


SimpleImputerPandas.X_to_numeric = X_to_numeric
SimpleImputerPandas.X_restore_categories = X_restore_categories


class KNNImputerPandas(KNNImputer):
    """
    KNNImputer() extended to maintain Pandas encapsulation
    and accept categorical along with numeric features.
    Output dtypes and categories should be as close as possible
    to input dtypes and categories.
    No object features allowed.
    Since KNN is sensitive to distance, StandardScaler is applied
    to input numeric features, and inverse scaling is done at the output.
    """

    def __init__(
        self,
        missing_values=-1.0,
        n_neighbors=5,
        weights='distance',
        metric='nan_euclidean',
        copy=True,
        add_indicator=False,
        keep_empty_features=False,
    ):
        super().__init__()
        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def fit(self, X, y=None):
        if self.missing_values != -1.0:
            print(f'warning: missing_values={self.missing_values}, but -1.0 is expected')
        X = X.copy()
        X_scaled = X.copy()
        self.X_dtypes = X.dtypes.to_dict()
        self.cat_cols = [c for c in X.select_dtypes(include='category')]
        self.num_cols = [c for c in X.select_dtypes(include='number')]
        super().set_output(transform='pandas')
        if len(self.num_cols) > 0:
            # KNN depends on distance, must standardize numeric columns
            self.ss = StandardScaler()
            self.ss.set_output(transform='pandas')
            # fit scaler on numeric features, and transform
            _ = self.ss.fit(X[self.num_cols])
            X_scaled[self.num_cols] = self.ss.transform(X[self.num_cols])
        # replace np.nan with the expected missing_values
        X_scaled_numeric = self.X_to_numeric(X_scaled).replace({np.nan: self.missing_values})
        # fit imputer on X
        return super().fit(X=X_scaled_numeric, y=y)

    def transform(self, X):
        # print('transform')
        X = X.copy()
        X_index = X.index
        X_scaled = X.copy()
        if len(self.num_cols) > 0:
            # transform numeric features with fitted scaler
            X_scaled[self.num_cols] = self.ss.transform(X[self.num_cols])
        # convert categorical to numeric, mark NaN with the missing_values variable
        X_scaled_numeric = self.X_to_numeric(X_scaled).replace({np.nan: self.missing_values})
        # apply fitted imputer to all features
        X_trans_scaled = super().transform(X_scaled_numeric)
        X_trans = X_trans_scaled.copy()
        if len(self.num_cols) > 0:
            # inverse scale (restore) numeric features
            X_trans[self.num_cols] = pd.DataFrame(
                self.ss.inverse_transform(X_trans_scaled[self.num_cols]),
                columns=self.num_cols,
                index=X_index,
            )
        # convert categorical columns back to categorical
        X_ret = self.X_restore_categories(X_trans, self.cat_cols, self.X_dtypes)
        # safeguard if dtypes are drifting
        for c in self.cat_cols:
            if X[c].dtype != X_ret[c].dtype:
                print(f'different dtypes {c}')
        return X_ret

    def fit_transform(self, X, y=None):
        _ = self.fit(X, y=None)
        return self.transform(X)

    def set_output(self, transform):
        pass


KNNImputerPandas.X_to_numeric = X_to_numeric
KNNImputerPandas.X_restore_categories = X_restore_categories
