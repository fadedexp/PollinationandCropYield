import pandas as pd


class Utils():
    def get_final_data(self, df):
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