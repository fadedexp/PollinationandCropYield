import pandas as pd
from sklearn import linear_model as lm
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class Utils():
    def __init__(self):
        pass

    def get_final_data(df):
        df['Row#'] = df['Row#'].replace({0 : 1})
        df['rain_intensity'] = df['AverageRainingDays'] / df['RainingDays']
        df['rain_variability'] = df['RainingDays'] / df['Row#']
        df['total_bees'] = df['honeybee'] + df['bumbles'] + df['andrena'] + df['osmia']
        df['clone_ratio'] = df['total_bees'] / df['clonesize']
        df['seed_ratio'] = df['seeds'] / df['fruitset']
        df['fruitmass_per_set'] = df['fruitmass'] / df['fruitset']
        df['honeybee_avgRD'] = df['honeybee'] * df['AverageRainingDays']
        df['clonesize_squared'] = df['clonesize'] * df['clonesize']
        df['fruitset_seeds'] = df['fruitset'] * df['seeds']
        df['fruitset_seeds_fruitmass'] = df['fruitset'] + df['seeds'] + df['fruitmass']
        df['sum_frs'] = (df['fruitmass'] + df['fruitset']) * df['seeds']
        df['fruitset_cat'] = pd.cut(df['fruitset'], bins=[0, 0.39, 0.45, 0.49, 0.52, 0.54, 0.56, 0.583, 0.6, 1], labels=list(range(9)))
        df['fruitmass_cat'] = pd.cut(df['fruitmass'], bins=[0.2, 0.39, 0.41, 0.43, 0.442, 0.448, 0.46, 0.48, 0.5, 0.7], labels=list(range(9)))
        df['seeds_cat'] = pd.qcut(df['seeds'], q=9, labels=list(range(9)))

        # # target_encoding
        # global_mean = df['yield'].mean()
        # alpha = 10
        # category_stats = df.groupby('fruitset_cat')['yield'].agg(['mean', 'count'])
        # category_stats['smoothed'] = (category_stats['mean'] * category_stats['count'] + global_mean * alpha) / (category_stats['count'] + alpha)
        # df['fruitset_encoded_smoothed'] = df['fruitset_cat'].map(category_stats['smoothed'])
        # alpha = 0.1
        # category_stats = df.groupby('seeds_cat')['yield'].agg(['mean', 'count'])
        # category_stats['smoothed'] = (category_stats['mean'] * category_stats['count'] + global_mean * alpha) / (category_stats['count'] + alpha)
        # df['seeds_encoded_smoothed'] = df['seeds_cat'].map(category_stats['smoothed'])
        # target_means = df.groupby('fruitmass_cat')['yield'].mean() 
        # df['fruitmass_encoded'] = df['fruitmass_cat'].map(target_means)


        drop_cols = [
            'id',
            'clonesize',
            'clonesize_squared',
            'MinOfUpperTRange', 
            'MaxOfLowerTRange', 
            'MinOfLowerTRange',
            'MaxOfUpperTRange',
            'AverageOfLowerTRange', 
            'AverageOfUpperTRange'
        ] 

        df = df.drop(columns=drop_cols)
        return df
    
    def get_final_model():
        seed = 42
        huber_params = {'epsilon': 1.4743037384391837,
                        'alpha': 0.0008997364985513577,
                        'fit_intercept': True,
                        'max_iter': 700,
                        'tol': 4.7218933129122393e-05,
                        'warm_start': False}
        pipeline_huber = Pipeline([
            ('Scaler', StandardScaler()),
            ('Model', lm.HuberRegressor(**huber_params))
        ])
        ridge_params = {'alpha': 0.00856065072895555, 'fit_intercept': True, 'solver': 'auto'}
        pipeline_ridge = Pipeline([
            ('Scaler', StandardScaler()),
            ('Model', lm.Ridge(**ridge_params))
        ])
        rf_params1 = {'n_estimators': 268,
        'max_depth': 7,
        'min_samples_split': 13,
        'min_samples_leaf': 10,
        'max_features': None,
        'max_samples': 0.5818250461230656,
        'min_impurity_decrease': 0.055,
        'min_weight_fraction_leaf': 0.0007226235667981045}
        pipeline_rf = Pipeline([
            ('Model', ensemble.RandomForestRegressor(**rf_params1, random_state=seed))
        ])
        pipeline_linear = Pipeline([
            ('Scaler', StandardScaler()),
            ('Model', lm.LinearRegression())
        ])
        elast_params = {'alpha': 5.699892707713835e-05,
        'l1_ratio': 0.9982856945861646,
        'max_iter': 5000,
        'tol': 2.5429621690102518e-05,
        'selection': 'random'}
        pipeline_elast = Pipeline([
            ('Scaler', StandardScaler()),
            ('Model', lm.ElasticNet(**elast_params))
        ])
        bagging_params = {'n_estimators': 156,
        'max_samples': 0.659621128654313,
        'max_features': 0.5846355752413999,
        'bootstrap': True,
        'bootstrap_features': True}
        pipeline_bagging = Pipeline([
            ('Model', ensemble.BaggingRegressor(**bagging_params, estimator=DecisionTreeRegressor(max_depth=7), random_state=seed))
        ])

        estimators = [
            ('rf', pipeline_rf),
            ('huber', pipeline_huber),
            ('ridge', pipeline_ridge),
            ('linear', pipeline_linear),
            ('bagging', pipeline_bagging),
            ('elast', pipeline_elast)
        ]
        stacking = ensemble.StackingRegressor(
            estimators=estimators,
            final_estimator=pipeline_rf,
            cv=5,
            n_jobs=-1,
            passthrough=True
        )
        return stacking