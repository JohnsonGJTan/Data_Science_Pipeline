from pathlib import Path

import datetime
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import missingno as msno
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from collections import defaultdict
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold


class EDAPipeline:
    '''
    Todo:
    - Make the subplots add dynamic
    - Add checks
    '''
    def __init__(self, data, experiment_name='', path='', verbose=False):
        
        self.verbose = verbose

        self.data = copy.deepcopy(data).convert_dtypes()
        self.nrows = data.shape[0]
        self.ncols = data.shape[1]

        self.feature_type = defaultdict()

        self.save= False
        if experiment_name != '' and path != '':
            self.save= True
            self.output_path = path + '/' + experiment_name + '/'
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            
    def set_feature_types(self, feature_type):
        
        self.feature_type = feature_type
        
        for feature in self.feature_type['categorical']:
            self.data[feature] = self.data[feature].astype('category')
        for feature in self.feature_type['discrete']:
            self.data[feature] = self.data[feature].astype('float')
        for feature in self.feature_type['continuous']:
            self.data[feature] = self.data[feature].astype('float')
    
        if self.verbose:
            print('Categorical features: ' + ', '.join([str(feature) for feature in self.feature_type['categorical']]))
            print('Discrete features: ' + ', '.join([str(feature) for feature in self.feature_type['discrete']]))
            print('Continuous features:' + ', '.join([str(feature) for feature in self.feature_type['continuous']]))

    def num_of_duplicates(self, verbose = False):
        count = sum(self.data.duplicated())
        if verbose or self.verbose:
            print("Number of duplicated rows: " + str(count))
        else:
            return count

    def missing_values(self, figsize=(24,24)):
        count = dict()
        for feature in self.feature_type['categorical'] + self.feature_type['discrete']+ self.feature_type['continuous']:
            count[feature] = self.data[feature].isna().sum()
        count['_total_'] = sum([True for _, row in self.data.iterrows() if any(row.isnull())])

        if count['_total_'] == 0:
            print('No missing values')
            return

        # Include distribution of missing data to determine MCAR, MAR, or MNAR
        fig, ax = plt.subplots(1,2,figsize=figsize, layout='constrained')
        fig.suptitle('Missing values density and correlation plots', fontsize='xx-large')
        msno.matrix(self.data, sparkline=False, ax=ax[0], labels=False)
        msno.heatmap(self.data, ax=ax[1])

        if self.save:
            fig.savefig(self.output_path + 'missing_values.png')

        if self.verbose:
            print("Number of missing values")
            table = pd.DataFrame({
                'Features': self.feature_type['categorical'] + self.feature_type['discrete'] + self.feature_type['continuous'] + ['Total'],
                'Percentage(%)': [100*count[feature]/self.data.shape[0] for feature in self.feature_type['categorical'] + self.feature_type['discrete'] + self.feature_type['continuous']] + [100*count['_total_']/self.data.shape[0]]
            })
            print(table.to_string(index=False))
        else:
            plt.close(fig)
            return count

    def plot_grid(self, features, title, plot, dim, figsize, **kwargs):
        fig = plt.figure(figsize=figsize, layout='constrained')
        fig.suptitle(title, size=figsize[1])
        subfigs = fig.subfigures(dim[0], dim[1])

        if dim[0] == 1 or dim[1] == 1:
            indexing = list(range(len(features)))
        else:
            indexing = list(product(range(dim[0]), range(dim[1])))[:len(features)]
        
        for index, feature in zip(indexing, features):
            if isinstance(index, int):
                subfig = subfigs[index]
            else:
                subfig = subfigs[index[0]][index[1]]
            plot(feature=feature, fig=subfig, **kwargs)

        if self.save:
            fig.savefig(self.output_path + title + '.png')
        
        if self.verbose:
            pass
        else:
            plt.close(fig)

    def distribution_plot(self, feature, group=None, fig=None):

        assert group == None or group in self.feature_type['categorical']

        if fig == None:
            fig, ax = plt.subplots(layout='constrained')
            fig.suptitle('Histogram for ' + feature)
        else:
            #assert isinstance(fig, Figure.subfigures), 'Invalid fig'
            ax = fig.add_subplot(111)

        discrete = feature in self.feature_type['discrete'] + self.feature_type['categorical']
        kde = feature in self.feature_type['discrete'] + self.feature_type['continuous']

        sns.histplot(ax=ax, data=self.data, x=feature, hue=group, discrete=discrete, multiple='stack', kde=kde)

        # create overall bar labesl
        if len(ax.containers[0]) < 15:
            overall_bar_height = [rect.get_y() + rect.get_height() for rect in ax.containers[-1]]
            ax.bar_label(ax.containers[-1], [f"{height/self.nrows:.2%}" for height in overall_bar_height])
            # create group bar labels
            if group != None and group != feature:
                
                group_bar_height = [rect.get_height() for rect in ax.containers[0]]
                ax.bar_label(
                    ax.containers[0],
                    [f"{(h_group/h_overall):.2%}" if h_overall >0 else '' for h_group, h_overall in zip(group_bar_height, overall_bar_height)]
                )



    def box_plot(self, feature, group=None, fig=None):
        
        assert group == None or group in self.feature_type['categorical'], 'invalid group.'
        assert feature in self.feature_type['discrete'] + self.feature_type['continuous'], 'feature is not of discrete or continuous type.'

        if fig== None:
            fig, ax = plt.subplots(layout='constrained')
            fig.suptitle('Boxplot for ' + feature)
        else:
            ax = fig.add_subplot(111)

        if group == None:
            sns.boxplot(ax=ax, data=self.data, y=feature)
        else:
            sns.boxplot(ax=ax, data=self.data, x=group, y=feature, hue=group)

    #def outlier_plot(self, feature, group: str, sigma=2, fig=None, include_base: bool = True):
    #    
    #    mean = self.data[feature].mean()
    #    std = self.data[feature].std()

    #    upper_bound = mean + sigma*std
    #    lower_bound = mean - sigma*std
    #    outlier_mask = [True if val > upper_bound or val < lower_bound else False for val in self.data[feature]]
    #    
    #    if fig == None:
    #        fig = plt.figure(layout='constrained')

    #    title = 'Distribution of ' + group + ' for outliers in ' + feature

    #    if sum(outlier_mask) == 0:
    #        ax = fig.add_subplot(111)
    #        fig.text(0.5,0.5, 'No outliers')
    #        ax.set_axis_off()
    #    else: 
    #        outlier_counts = self.data.loc[outlier_mask,group].value_counts().sort_index()
    #        
    #        if include_base:
    #            ax_outlier = fig.add_subplot(1,2,1)
    #        else:
    #            ax_outlier = fig.add_subplot(1,1,1)

    #        sns.barplot(x=outlier_counts.index, y=outlier_counts.values, ax=ax_outlier)
    #        ax_outlier.bar_label(ax_outlier.containers[0], labels=[f"{x/sum(outlier_counts.values):.2%}" for x in outlier_counts.values])

    #        if include_base:
    #            title += ' (L) vs not outliers (R)'
    #            not_outlier_counts = self.data[group].value_counts().sort_index()
    #            ax_base = fig.add_subplot(1,2,2)
    #            sns.barplot(x=not_outlier_counts.index, y=not_outlier_counts.values, ax=ax_base)
    #            ax_base.bar_label(ax_base.containers[0], labels=[f"{x/sum(not_outlier_counts.values):.2%}" for x in not_outlier_counts.values])

    #    fig.suptitle(title, size='xx-large')

    #    if self.save:
    #        fig.savefig(self.output_path + 'outlier_' + feature + '_distribution_plots.png')

    #    if self.verbose:
    #        count = sum(outlier_mask)
    #        print(str(count) + ' outliers at sigma=' + str(sigma) + ' for ' + feature + '. Bounds given by [' + str(lower_bound) + ',' + str(upper_bound) + '].')
    #    else:
    #        plt.close(fig)
    #        return (lower_bound, upper_bound), outlier_mask

    def outlier_plot(self, feature, group: str, sigma: list[int] = [2], fig=None, include_base: bool = True):
        
        assert len(set(self.data[group])) < 10, 'group has too many groups!'

        mean = self.data[feature].mean()
        std = self.data[feature].std()

        data_long = pd.DataFrame(
            {
                group: [],
                'count': [],
                'sigma': [],
            }
            
        )

        # compute outlier masks for the various sigmas
        for sig in sigma:
            ub, lb = mean + sig*std, mean - sig*std
            outlier_mask = [True if val >= ub or val <= lb else False for val in self.data[feature]]

            if sum(outlier_mask) == 0:
                break

            group_distribution = pd.DataFrame(self.data.loc[outlier_mask, group].value_counts()/self.data.loc[outlier_mask, group].value_counts().sum()).reset_index()
            group_distribution['sigma'] = sig

            if data_long.empty:
                data_long = group_distribution
            else:
                data_long = pd.concat([data_long, group_distribution])
        
        data_long.columns = [group, 'percentage', 'sigma']

        if fig == None:
            fig = plt.figure(layout='constrained')
        
        fig.suptitle('Distribution of ' + group + ' for outliers in ' + feature + '.', size='xx-large')
        ax = fig.add_subplot(111)
        ax.set_ylim(0,1)
        sns.barplot(
            data=data_long,
            x='sigma',
            y='percentage',
            hue=group,
            ax=ax
        )


    def KS_test(self, new_EDA):
        # Find intersection of columns
        features = set(self.feature_type['discrete'] + self.feature_type['continuous']) & set(new_EDA.feature_type['discrete'] + new_EDA.feature_type['continuous'])
        
        ks_statistic = dict()

        for feature in features:
            training = self.data[feature]
            testing = new_EDA.data[feature]
            if feature in self.feature_type['discrete'] or self.feature_type['continuous']:
                ks_statistic[feature] = stats.kstest(rvs=training.dropna(),cdf=testing.dropna())
            #if feature in self.feature_type['discrete'] or self.feature_type['categorical']:
            #    training_count = training.value_counts().sort_index().to_numpy()
            #    testing_count= testing.value_counts().sort_index().to_numpy()
            #    cs_statistic = stats.chi2_contingency(np.array([training_count, testing_count]))
            #    print(feature + ": " + str(cs_statistic.pvalue) + "(Chi-squared)")
        
        if self.verbose:
            print('Kolmogorov-Smirnov statistic (p-value) for train-test drift')
            table = pd.DataFrame({
                'Feature': [feature for feature,_ in ks_statistic.items()],
                'p-value': [statistic.pvalue for _, statistic in ks_statistic.items()]
            })
            print(table.to_string(index=False))
            #for feature, statistic in ks_statistic.items():
            #    print(feature + ": " + str(statistic.pvalue))
        else:
            return ks_statistic

    #def heat_maps(self, figsize, target=None, value=None, verbose=False):
    #    
    #    cmap = cm.get_cmap('magma')
    #    fig = plt.figure(1, figsize=figsize, layout='constrained')
    #    title = 'Heatmap of distribution along two-variables'

    #    features = self.feature_type['discrete'] + self.feature_type['continuous']

    #    # Compute pivot tables and track largest frequency
    #    max_freq = 0
    #    pivot = dict()
    #    for i in range(1,len(features)):
    #        for j in range(i):
    #            pivot[frozenset([features[i], features[j]])] = self.data.groupby(features[i])[features[j]].value_counts(dropna=True).unstack(fill_value=0).astype(int)
    #            max_freq = max(max_freq, pivot[frozenset([features[i], features[j]])].to_numpy().max())

    #    if target != None and value != None:
    #        assert target in self.feature_type['categorical'] and value in self.data[target].unique(), 'target not a categorical variable or value is not a possible category.'
    #        
    #        for i in range(1, len(features)):
    #            for j in range(i):
    #                value_pivot = self.data.groupby([features[i], features[j]])[target].value_counts().unstack(fill_value=0).loc[:,value].unstack(fill_value=0)
    #                pivot[frozenset([features[i], features[j]])] = 100*value_pivot.div(pivot[frozenset([features[i], features[j]])] + 1.e-17)

    #        max_freq = 100
    #        title += ' (' + value + ' proportions)'

    #    # Plot heat maps along upper left triangle
    #    fig.suptitle(title, size='xx-large')

    #    n = len(features) - 1
    #    for i in range(1, n + 1):
    #        for j in range(i):
    #            ax = fig.add_subplot(n, n, (i - 1)*n + (j + 1))
    #            sns.heatmap(ax=ax, data=pivot[frozenset([features[i], features[j]])],
    #                cbar=False, vmax=max_freq, vmin=0, cmap=cmap, annot=True, fmt='.0f')
    #            ax.invert_yaxis()

    #            if i < n:
    #                ax.tick_params(axis='x', which='both', bottom=False,top=False,labelbottom=False)
    #                ax.xaxis.set_visible(False)
    #            else:
    #                ax.tick_params(axis='x', labelrotation=90)
    #            if j > 0:
    #                ax.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
    #                ax.yaxis.set_visible(False)
    #            else:
    #                ax.tick_params(axis='y', labelrotation=0)
    #                

    #    # Shared colorbar
    #    normalizer = Normalize(0, max_freq)
    #    im = cm.ScalarMappable(norm=normalizer)
    #    fig.colorbar(im, ax=fig.axes)

    #    if self.save:
    #        fig.savefig(self.output_path + 'correlation_heat_maps.png')

    #    if self.verbose:
    #        pass
    #    else:
    #        plt.close(fig)
    #        return pivot

class PreprocessingPipeline:
    '''
    Todo
    - Add scale/normalize method
    - Add class imbalance feature, e.g. SMOTE, stratified k-fold CV
    - Add Feature Engineering, e.g. interaction, binned, polynomial
    - Add checks
    '''    
    def __init__(self, train_raw: pd.DataFrame, feature_type: dict, test_raw: pd.DataFrame, target: str, target_type: str):
        
        self.train_raw = train_raw.convert_dtypes()        
        self.test_raw = test_raw.convert_dtypes()
        
        self.feature_type = {
            'categorical': feature_type['categorical'],
            'discrete': feature_type['discrete'],
            'continuous': feature_type['continuous'],
        }
        self.target = target

        self.target_type = target_type
        self.pipeline = []
        '''
        An object in the pipeline shoud be a dictionary with the following items:
        label: name of pipe
        description: text description of what pipe does
        function: the function which manipulates data
        arguments: a dictionary of arguments for said function 
        '''
            
    def preprocess(self, verbose: bool = False, file_name: str = ''):

        train = self.train_raw.copy(deep=True)
        test = self.test_raw.copy(deep=True)

        # Set column types
        for feature in self.feature_type['categorical']:
            train[feature] = train[feature].astype('category')
            test[feature] = test[feature].astype('category')
        for feature in self.feature_type['discrete']:
            train[feature] = train[feature].astype('float')
            test[feature] = test[feature].astype('float')
        for feature in self.feature_type['continuous']:
            train[feature] = train[feature].astype('float')
            test[feature] = test[feature].astype('float')

        assert self.target_type == ('category' or 'float')
        train[self.target] = train[self.target].astype(self.target_type)

        for description, func, kwargs in self.pipeline:
            train, test = func(train, test, **kwargs)
            if verbose:
                print(description)

        if file_name != '':
            with open(file_name + '.pkl', 'wb') as f:
                pickle.dump((train,test), f)

        return train, test

    @staticmethod
    def static_replace_col(data: pd.DataFrame, col_name: str, new_col):
        return pd.concat(
            [
                data.drop(col_name, axis=1),
                pd.Series(new_col, name=col_name)
            ],
            axis=1
        )

    def replace_col(self, col_name: str, new_col):
        self.pipeline.append(
            (
                '',
                self.static_replace_col,
                {
                    'col_name': col_name,
                    'new_col': new_col
                }
            )
        )
    

    ### Phase 1: Data Imputation

    @staticmethod
    def static_drop_col(train: pd.DataFrame, test: pd.DataFrame, col_name: str):
        return train.drop(col_name, axis=1, errors='ignore'), test.drop(col_name, axis=1, errors='ignore')
    
    def drop_col(self, col_name):
        self.pipeline.append(
            (
                'Dropped column: ' + col_name,
                self.static_drop_col,
                {
                    'col_name': col_name
                }
            )
        )
    
    @staticmethod
    def static_missing_indicator(train: pd.DataFrame, test: pd.DataFrame, col_name: str):
        missing_indicator_train = train[col_name].isnull()
        missing_indicator_test= test[col_name].isnull()
        missing_indicator_test.name = col_name + '_missing'
        missing_indicator_train.name = col_name + '_missing'

        return pd.concat([train, missing_indicator_train], axis=1), pd.concat([test, missing_indicator_test], axis=1)

    def missing_indicator(self, col_name):
        self.pipeline.append(
            (
                'Missing indicator appended: ' + col_name,
                self.static_missing_indicator,
                {
                    'col_name': col_name
                }
            )
        )
        
    @staticmethod
    def static_simple_categorical_impute(train: pd.DataFrame, test: pd.DataFrame, col_name: str, fill: str = '_mode_'):
        
        if fill == '_mode_':
            value = train[col_name].mode()[0]
        else:
            if fill not in train[col_name].cat.categories:
                train[col_name] = train[col_name].cat.add_categories([fill])
                test[col_name] = test[col_name].cat.add_categories([fill])
            value = fill
        impute_col_train = train[col_name].fillna(value)
        impute_col_test = test[col_name].fillna(value)
        return PreprocessingPipeline.static_replace_col(train, col_name, impute_col_train), PreprocessingPipeline.static_replace_col(test, col_name, impute_col_test) 

    def simple_categorical_impute(self, col_name, fill: str = '_mode_'):
        self.pipeline.append(
            (
                'Missing values in ' + col_name + ' imputed with value ' + fill,
                self.static_simple_categorical_impute,
                {
                    'col_name': col_name,
                    'fill': fill
                }
            )
        )
    
    @staticmethod
    def static_median_impute(train: pd.DataFrame, test: pd.DataFrame, col_name):
        median = train[col_name].median()
        impute_col_train = train[col_name].fillna(median)
        impute_col_test = test[col_name].fillna(median)
        return PreprocessingPipeline.static_replace_col(train, col_name, impute_col_train), PreprocessingPipeline.static_replace_col(test, col_name, impute_col_test)
    
    def median_impute(self, col_name):
        self.pipeline.append(
            (
                'Missing values in ' + col_name + ' imputed with median',
                self.static_median_impute,
                {
                    'col_name': col_name
                }
            )
        )

    ### Phase 2: Outlier Preprocessing

    @staticmethod
    def _outlier_preprocessing(
        train: pd.DataFrame,
        test: pd.DataFrame,
        col_name: str,
        outlier_levels: list[int],
        impute_method: str,
        #bounds= []
    ):
        methods = ['clip', 'remove', 'none']
        assert impute_method in methods, 'impute method unknown'
        assert sum(train[col_name].isna()) + sum(test[col_name].isna()) == 0, 'column contains missing values which need to be imputed.'
        assert outlier_levels == sorted(outlier_levels), 'sigmas are not sorted'
        assert outlier_levels[0] > 0, 'smallest sigma must be positive'

        mean = train[col_name].mean()
        std = train[col_name].std()

        # compute outlier mask level
        outlier_mask_train = [0] * train.shape[0]
        outlier_mask_test = [0] * test.shape[0]

        for sigma in outlier_levels:
            ub = mean + sigma * std
            lb = mean - sigma * std
            sigma_mask_train = [1 if val > ub else -1 if val < lb else 0 for val in train[col_name]]
            sigma_mask_test = [1 if val > ub else -1 if val < lb else 0 for val in test[col_name]]

            outlier_mask_train = [a + b for a, b in zip(outlier_mask_train, sigma_mask_train)]
            outlier_mask_test = [a + b for a, b in zip(outlier_mask_test, sigma_mask_test)]

        lub = mean + outlier_levels[0] * std
        glb = mean - outlier_levels[0] * std

        if impute_method == 'remove':
            return train.drop([1 if val > 0 else 0 for val in outlier_mask_train]), test
        elif impute_method == 'clip':
            outlier_impute_train = train[col_name].clip(lower=glb, upper=lub)
            outlier_impute_test = test[col_name].clip(lower=glb, upper=lub)
            return pd.concat(
                [
                    PreprocessingPipeline.static_replace_col(train, col_name, outlier_impute_train),
                    pd.Series(outlier_mask_train, name=col_name + '_outlier_mask')
                ],
                axis=1
            ), pd.concat(
                [
                    PreprocessingPipeline.static_replace_col(test, col_name, outlier_impute_test),
                    pd.Series(outlier_mask_test, name=col_name + '_outlier_mask')
                ],
                axis=1
            )
        elif impute_method == 'none':
            return pd.concat(
                [
                    train,
                    pd.Series(outlier_mask_train, name=col_name + '_outlier_mask')
                ],
                axis=1
            ), pd.concat(
                [
                    test,
                    pd.Series(outlier_mask_test, name=col_name + '_outlier_mask')
                ],
                axis=1
            )

    def outlier_preprocessing(self, col_name, outlier_levels, impute_method = 'clip'):
        self.pipeline.append(
            (
                '',
                self._outlier_preprocessing,
                {
                    'col_name': col_name,
                    'outlier_levels': outlier_levels,
                    'impute_method': impute_method,
                }
            )
        )

    ### Phase 3: Feature Engineering
    '''
    - Label Encoding for target variable
    - One Hot Encoding for features with low class count
    - dense vector embedding for features with high class count
    - Ordinal encoding for ordered categorical/discrete features
    '''

    @staticmethod
    def static_ordinal_encoding(train: pd.DataFrame, test: pd.DataFrame, col_names: list[str]):
        ordinal_encoder = OrdinalEncoder()
        ordinal_encoded = ordinal_encoder.fit(train[col_names])
        ordinal_encoded.set_output(transform='pandas')
        ordinal_encoded_train = ordinal_encoded.transform(train[col_names])
        ordinal_encoded_test = ordinal_encoded.transform(test[col_names])

        assert isinstance(ordinal_encoded_train, pd.DataFrame)
        assert isinstance(ordinal_encoded_test, pd.DataFrame)

        return (
            pd.concat([train.drop(col_names, axis=1), ordinal_encoded_train],axis=1),
            pd.concat([test.drop(col_names, axis=1), ordinal_encoded_test],axis=1),
        )

    def ordinal_encoding(self, col_names):
        self.pipeline.append(
            (
                'Ordinal encoding applied to columns ' + ', '.join(col_names) + '.',
                self.static_ordinal_encoding,
                {
                    'col_names': col_names
                }
            )
        )
    
        
    @staticmethod
    def static_OH_encoding(train: pd.DataFrame, test: pd.DataFrame, col_names: list[str]):
        OHEncoder = OneHotEncoder(sparse_output=False, drop='if_binary')
        one_hot_encoded = OHEncoder.fit(train[col_names])
        one_hot_encoded.set_output(transform='pandas')
        one_hot_encoded_train = one_hot_encoded.transform(train[col_names])
        one_hot_encoded_test = one_hot_encoded.transform(test[col_names])

        assert isinstance(one_hot_encoded_train, pd.DataFrame)
        assert isinstance(one_hot_encoded_test, pd.DataFrame)

        return (
            pd.concat([train.drop(col_names, axis=1), one_hot_encoded_train],axis=1),
            pd.concat([test.drop(col_names, axis=1), one_hot_encoded_test],axis=1),
        )
        
    def OH_encoding(self, col_names):
        self.pipeline.append(
            (
                'One-hot encoding applied to columns ' + ', '.join(col_names) + '.',
                self.static_OH_encoding,
                {
                    'col_names': col_names
                }
            )
        )

    @staticmethod
    def static_label_encoding(train: pd.DataFrame, test: pd.DataFrame, col_name: str, true_test: bool = False):
        LEncoder = LabelEncoder()
        LEncoder.fit(train[col_name])
        label_encoded_train = LEncoder.transform(train[col_name])
        
        if true_test:
            return PreprocessingPipeline.static_replace_col(train, col_name=col_name, new_col=label_encoded_train), test 
        else:
            label_encoded_test = LEncoder.transform(test[col_name])
            return (
                PreprocessingPipeline.static_replace_col(train, col_name=col_name, new_col=label_encoded_train), 
                PreprocessingPipeline.static_replace_col(test, col_name=col_name, new_col=label_encoded_test)
            )
        

    def label_encoding(self, col_name: str, true_test: bool = False):
        self.pipeline.append(
            (
                'Label encoding applied to column ' + col_name,
                self.static_label_encoding,
                {
                    'col_name': col_name,
                    'true_test': true_test
                }
            )
        )

    @staticmethod
    def weighted_mean(data:pd.DataFrame, cols: list[str], target: str, overall_weight: float = 0):
        
        option_mean = data.groupby(cols)[target].mean()
        option_weight = data.groupby(cols).size()
        overall_mean = data[target].mean()

        return (option_mean * option_weight + overall_mean * overall_weight) / (option_weight + overall_weight)

    @staticmethod
    def static_target_encoding(train: pd.DataFrame, test: pd.DataFrame, label: str, col_names: list[str], target:str, 
                               overall_weight: float = 0, k: int = 1, drop: bool = False):
        
        assert k > 0, 'cv needs to be a positive integer.'

        if k == 1:
            weighted_mean = PreprocessingPipeline.weighted_mean(train, col_names, target, overall_weight)
            output_train = train.merge(weighted_mean.rename(label), on=col_names)        
            output_test = test.merge(weighted_mean.rename(label), on=col_names)
        else:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            k_fold_splits_train = []
            k_fold_weights = []
            for train_index, test_index in kf.split(train):
                weighted_mean = PreprocessingPipeline.weighted_mean(train.iloc[train_index], col_names, target, overall_weight)
                k_fold_splits_train.append(train.iloc[test_index].merge(weighted_mean.rename(label), on=col_names))
                k_fold_weights.append(weighted_mean)
            output_train = pd.concat(k_fold_splits_train)
            weights_avg = sum(k_fold_weights)/k

            assert isinstance(weights_avg, pd.Series)
            output_test = test.merge(weights_avg.rename(label), on=col_names)
        if drop:
            output_train = output_train.drop(col_names, axis=1)
            output_test = output_test.drop(col_names, axis=1)

        return output_train, output_test

    def target_encoding(self, label: str, col_names: list[str], target:str, 
                        overall_weight: float = 0, k: int = 1, drop: bool = False):
        self.pipeline.append(
            (
                'Target encoding ' + ', '.join(col_names) + '.',
                self.static_target_encoding,
                {
                    'label': label,
                    'col_names': col_names,
                    'target': target,
                    'overall_weight': overall_weight,
                    'k': k,
                    'drop': drop
                }
            )
        )

    @staticmethod
    def static_polynomial_engineering(train: pd.DataFrame, test: pd.DataFrame, col_names: list[str], degree: int):
        pass

    def polynomial_engineering(self, train: pd.DataFrame, test: pd.DataFrame, col_names: list[str], degree: int):
        self.pipeline.append(
            (
                '',
                self.static_polynomial_engineering,
                {
                    'col_names': col_names,
                    'degree': degree
                }
            )
        )

    @staticmethod
    def ymd_to_time(year: list[int], month: list[int], day: list[int]):
        assert max(month) <= 12 and min(month) > 0, 'There are months outside range.' 
        assert max(day) <= 31 and min(day) > 0, 'There are days outside range.' 

        return [datetime.datetime(y, m, d) for y,m,d in zip(year, month, day)]

class ModelingPipeline:

    def __init__(self, train: pd.DataFrame, target: str, test: pd.DataFrame = pd.DataFrame([]), verbose: bool = False):
        
        assert target in train.columns, 'target variable is not in data'

        self.verbose = verbose
        self.pipeline = []
        '''
        - model name
        - classifier
        - parameters for classifier
        '''
        
        if test.empty:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                train.drop(axis=1,labels=target),
                train[target],
                test_size=0.2,
                random_state=42,
                stratify=train[target]
            )
        else:
            self.X_train, self.y_train = train.drop(target, axis=1, errors='ignore'), train[target]
            self.X_test, self.y_test = test.drop(target, axis=1, errors='ignore'), test[target]

        self.trained_models = defaultdict()

    def train_model(self, model, params, n_iter: int = 20, cv: int = 5, scoring='f1_weighted', n_jobs=-1):
        classifier = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=42
        )
        classifier.fit(self.X_train, self.y_train)
        return classifier

    def process_pipeline(self, n_iter: int = 20, cv: int = 5, scoring = 'f1_weighted', n_jobs=-1):
        
        self.trained_models['best'] = {
            'model_name': '',
            'classifier': None,
            'train_score': 0,
            'test_score': 0
        }
        for model_name, classifier, params in self.pipeline:
            classifier = self.train_model(classifier, params, n_iter, cv, scoring, n_jobs)
            y_pred = classifier.predict(self.X_test)
            test_score = sum(y_pred == self.y_test) / len(y_pred)
            self.trained_models[model_name] = {
                'model_name': model_name,
                'classifier': classifier.best_estimator_,
                'train_score': classifier.best_score_,
                'test_score': test_score
            }
            if self.trained_models[model_name]['train_score'] > self.trained_models['best']['train_score']:
                self.trained_models['best'] = copy.deepcopy(self.trained_models[model_name])