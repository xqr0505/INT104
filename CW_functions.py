# pip install numpy pandas matplotlib seaborn scikit-learn scipy itertools-s tabulate
# pip install openpyxl
# pip install scipy scikit-learn pandas matplotlib seaborn numpy
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.externals._scipy.sparse.csgraph import laplacian
from sklearn.feature_selection import mutual_info_classif
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, pairwise_distances, \
    ConfusionMatrixDisplay, confusion_matrix, adjusted_rand_score, silhouette_score, davies_bouldin_score, \
    classification_report
from sklearn.feature_selection import RFECV
from sklearn.model_selection import learning_curve,cross_val_score, StratifiedKFold,GridSearchCV,ParameterGrid,cross_validate
import itertools
from itertools import combinations, product
from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import hamming
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

class FeatureEngineering:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def create_statistical_features(self, df, question_cols):
        """Creates statistical summary features from question scores."""
        df_out = df.copy()
        stat_features = []
        try:
            numeric_qs = df_out[question_cols].select_dtypes(include=np.number)
            if not numeric_qs.empty:
                median_scores = numeric_qs.median(axis=0)
                df_out['Std_Dev'] = numeric_qs.std(axis=1)
                df_out['Total_Score'] = numeric_qs.sum(axis=1)
                df_out['Max_Score'] = numeric_qs.max(axis=1)
                df_out['Min_Score'] = numeric_qs.min(axis=1)
                df_out['Score_Range'] = df_out['Max_Score'] - df_out['Min_Score']
                df_out['HighScore_Count'] = (numeric_qs > median_scores[numeric_qs.columns]).sum(axis=1)
                stat_features = ['Std_Dev', 'Total_Score', 'Max_Score', 'Min_Score', 'Score_Range', 'HighScore_Count']
            else:
                print("Warning: No numeric question columns found for statistical feature generation.")
        except Exception as e:
            print(f"Error creating statistical features: {e}")
        return df_out, stat_features

    def create_interaction_features(self, df, question_cols):
        """Creates pairwise interaction features (sum, diff, product, ratio)."""
        df_out = df.copy()
        interact_features = []
        numeric_qs = df_out[question_cols].select_dtypes(include=np.number)
        if numeric_qs.empty:
            print("Warning: No numeric question columns found for interaction feature generation.")
            return df_out, []
        for q1, q2 in combinations(numeric_qs.columns, 2):
            if q1 in df_out.columns and q2 in df_out.columns:
                add_feat_name = f'{q1}_plus_{q2}'
                df_out[add_feat_name] = df_out[q1] + df_out[q2]
                interact_features.append(add_feat_name)
                diff_feat_name = f'{q1}_absdiff_{q2}'
                df_out[diff_feat_name] = abs(df_out[q1] - df_out[q2])
                interact_features.append(diff_feat_name)
                times_feat_name = f'{q1}_times_{q2}'
                df_out[times_feat_name] = np.log1p(df_out[q1] * df_out[q2])
                interact_features.append(times_feat_name)
                div_feat_name = f'{q1}_div_{q2}'
                df_out[div_feat_name] = np.log1p(df_out[q1] / (df_out[q2] + 1e-6))
                interact_features.append(div_feat_name)
                div_rev_feat_name = f'{q2}_div_{q1}'
                df_out[div_rev_feat_name] = np.log1p(df_out[q2] / (df_out[q1] + 1e-6))
                interact_features.append(div_rev_feat_name)
        return df_out, interact_features

    def filter_correlated_features(self, df, new_features, original_features, corr_threshold=0.8):
        """Filters newly created features based on correlation while retaining original features."""
        if not new_features:
            return df, []

        # Combine new features and original features for correlation calculation
        all_features = new_features + original_features
        corr_matrix = df[all_features].corr().abs()

        # Exclude original features from being dropped
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in new_features if any(upper[col][new_features] > corr_threshold)]

        # Drop only the new features with high correlation
        df_filtered = df.drop(columns=to_drop)
        final_features = [f for f in new_features if f not in to_drop] + original_features

        print(f"Removed {len(to_drop)} highly correlated new features (Threshold > {corr_threshold}).")
        return df_filtered, final_features

    def rank_features(self, df, features_to_rank, target_col):
        """Ranks features using Chi2 and Mutual Information."""
        X = df[features_to_rank].copy()
        y = df[target_col]
        numeric_cols = X.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            X[numeric_cols] = X[numeric_cols].apply(lambda x: x - x.min() + 1e-6 if x.min() <= 0 else x, axis=0)
        X = X.select_dtypes(include=np.number)
        valid_features = X.columns.tolist()
        if not valid_features:
            print("Error: No valid numeric features found for ranking.")
            return pd.DataFrame(columns=['Feature', 'Chi2_Value', 'MI_Score', 'Composite_Score'])
        try:
            chi2_values, _ = chi2(X, y)
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            results = pd.DataFrame({
                'Feature': valid_features,
                'Chi2_Value': chi2_values,
                'MI_Score': mi_scores
            })
            scaler = MinMaxScaler()
            if results['Chi2_Value'].nunique() > 1:
                results['Chi2_Normalized'] = scaler.fit_transform(results[['Chi2_Value']])
            else:
                results['Chi2_Normalized'] = 0.5
            if results['MI_Score'].nunique() > 1:
                results['MI_Normalized'] = scaler.fit_transform(results[['MI_Score']])
            else:
                results['MI_Normalized'] = 0.5
            results['Composite_Score'] = (results['Chi2_Normalized'] + results['MI_Normalized']) / 2
            final_ranking = results.sort_values('Composite_Score', ascending=False)[
                ['Feature', 'Chi2_Value', 'MI_Score', 'Composite_Score']
            ].reset_index(drop=True)
            return final_ranking
        except Exception as e:
            print(f"Error during feature ranking: {e}")
            return pd.DataFrame(columns=['Feature', 'Chi2_Value', 'MI_Score', 'Composite_Score'])

    def chi2_feature_ranking(self, df, features, target_col):
        """Ranks features based on Chi-square test for feature selection."""
        X = df[features].copy()
        y = df[target_col]

        # Prepare data for chi2 (must be non-negative)
        numeric_cols = X.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            X[numeric_cols] = X[numeric_cols].apply(lambda x: x - x.min() + 1e-6 if x.min() <= 0 else x, axis=0)

        # Filter to numeric features only
        X = X.select_dtypes(include=np.number)
        valid_features = X.columns.tolist()

        if not valid_features:
            print("Error: No valid numeric features found for chi-square ranking.")
            return pd.DataFrame(columns=['Feature', 'Chi2_Value', 'p_value'])

        try:
            # Calculate chi2 values and p-values
            chi2_values, p_values = chi2(X, y)

            # Create and sort results
            results = pd.DataFrame({
                'Feature': valid_features,
                'Chi2_Value': chi2_values,
                'p_value': p_values
            })

            # Sort by chi2 value in descending order
            sorted_results = results.sort_values('Chi2_Value', ascending=False).reset_index(drop=True)
            return sorted_results

        except Exception as e:
            print(f"Error during chi-square feature ranking: {e}")
            return pd.DataFrame(columns=['Feature', 'Chi2_Value', 'p_value'])

    def mutual_info_feature_ranking(self, df, features, target_col):
        """Ranks features based on Mutual Information for feature selection."""
        X = df[features].copy()
        y = df[target_col]

        # Filter to numeric features only
        X = X.select_dtypes(include=np.number)
        valid_features = X.columns.tolist()

        if not valid_features:
            print("Error: No valid numeric features found for mutual information ranking.")
            return pd.DataFrame(columns=['Feature', 'MI_Score'])

        try:
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)

            # Create and sort results
            results = pd.DataFrame({
                'Feature': valid_features,
                'MI_Score': mi_scores
            })

            # Sort by MI score in descending order
            sorted_results = results.sort_values('MI_Score', ascending=False).reset_index(drop=True)
            return sorted_results

        except Exception as e:
            print(f"Error during mutual information feature ranking: {e}")
            return pd.DataFrame(columns=['Feature', 'MI_Score'])

    def perform_rfecv_with_logistic_regression(self, data, target_col):
        """Performs RFECV using Logistic Regression with optimized hyperparameters."""
        X = data.drop(columns=[target_col])  # Use ALL features except target
        y = data[target_col]
        # Hyperparameter tuning (optional but recommended)
        param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
        grid_search = GridSearchCV(
            LogisticRegression(random_state=self.random_state),
            param_grid,
            cv=5,
            scoring='accuracy'
        )
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        # RFECV with best params
        estimator = LogisticRegression(**best_params, random_state=self.random_state)
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=StratifiedKFold(5),
            scoring=make_scorer(accuracy_score),
            min_features_to_select=3  # Minimum number of features to keep
        )
        rfecv.fit(X, y)
        # Get selected features
        selected_features = X.columns[rfecv.support_].tolist()
        optimal_num_features = rfecv.n_features_
        return selected_features, optimal_num_features

    def perform_sequential_forward_selection(self, X, y):
        """Perform feature selection using Sequential Forward Selection with different base models"""


        print("\n=== Performing Feature Selection with SFS ===")

        # 1. GaussianNB feature selection
        gnb = GaussianNB()
        sfs_gnb = SequentialFeatureSelector(
            estimator=gnb,
            n_features_to_select='auto',
            direction='forward',
            scoring='accuracy',
            cv=5
        )
        sfs_gnb.fit(X, y)
        gnb_features = X.columns[sfs_gnb.get_support()].tolist()

        # 2. KNN feature selection (with pipeline)
        knn_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])
        sfs_knn = SequentialFeatureSelector(
            estimator=knn_pipe,
            n_features_to_select='auto',
            direction='forward',
            scoring='accuracy',
            cv=5
        )
        sfs_knn.fit(X, y)
        knn_features = X.columns[sfs_knn.get_support()].tolist()

        # 3. Decision Tree feature selection
        dt = DecisionTreeClassifier(random_state=self.random_state)
        sfs_dt = SequentialFeatureSelector(
            estimator=dt,
            n_features_to_select='auto',
            direction='forward',
            scoring='accuracy',
            cv=5
        )
        sfs_dt.fit(X, y)
        dt_features = X.columns[sfs_dt.get_support()].tolist()

        # 4. Random Forest feature selection
        rf = RandomForestClassifier(random_state=self.random_state)
        sfs_rf = SequentialFeatureSelector(
            estimator=rf,
            n_features_to_select='auto',
            direction='forward',
            scoring='accuracy',
            cv=5
        )
        sfs_rf.fit(X, y)
        rf_features = X.columns[sfs_rf.get_support()].tolist()

        return {
            'GaussianNB': gnb_features,
            'KNN': knn_features,
            'DecisionTree': dt_features,
            'RandomForest': rf_features
        }

    def similarity_based_feature_selection(self, df, target_col, top_k):
        """Selects top-k features based on similarity with a distance matrix."""
        # Check if target column exists in dataframe before dropping
        X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
        y = df[target_col] if target_col in df.columns else None
        distance_matrix = pairwise_distances(X, metric='euclidean')

        def similarity_score(feature, distance_matrix):
            return np.corrcoef(feature, distance_matrix.mean(axis=1))[0, 1]

        scores = {col: similarity_score(X[col], distance_matrix) for col in X.columns}
        top_features = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return df[top_features], top_features

    def laplacian_score_feature_selection(self, df, top_k):
        """Selects top-k features based on Laplacian Score."""
        X = df.select_dtypes(include=np.number)
        laplacian_matrix = laplacian(pairwise_distances(X, metric='euclidean'), normed=True)
        scores = np.sum((X.T @ laplacian_matrix) * X.T, axis=1)
        top_features = X.columns[np.argsort(scores)[:top_k]].tolist()
        return df[top_features], top_features

    def fisher_score_feature_selection(self, data, target_col=None, top_k=15):
        """
        使用 Fisher Score 进行特征选择，选择分类能力最强的 top_k 个特征。

        Parameters:
            data (pd.DataFrame): 包含特征的数据框（可以包含或不包含目标列）
            target_col (str): 目标列的名称，如果为 None 则假设数据不包含目标列
            top_k (int): 要选择的顶部特征数量

        Returns:
            tuple: (选择后的数据框, 选择的特征列表)
        """
        # 检查数据是否包含目标列
        if target_col is not None:
            if target_col not in data.columns:
                raise ValueError(f"目标列 {target_col} 不在数据框中")
            X = data.drop(columns=[target_col])
            y = data[target_col]
        else:
            # 假设最后一列是目标列
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

        feature_names = X.columns.tolist()

        # 计算每个特征的 Fisher Score
        fisher_scores = []
        classes = y.unique()

        for feature in feature_names:
            numerator = 0
            denominator = 0
            overall_mean = X[feature].mean()

            for c in classes:
                # 获取当前类别的样本
                class_samples = X.loc[y == c, feature]

                # 计算类内均值
                class_mean = class_samples.mean()

                # 计算类内方差
                class_var = class_samples.var() if len(class_samples) > 1 else 0

                # 该类别样本数量
                n_c = len(class_samples)

                # Fisher Score 计算的分子部分（类间方差）
                numerator += n_c * ((class_mean - overall_mean) ** 2)

                # Fisher Score 计算的分母部分（类内方差）
                denominator += n_c * class_var

            # 避免除以零
            if denominator == 0:
                fisher_score = 0
            else:
                fisher_score = numerator / denominator

            fisher_scores.append((feature, fisher_score))

        # 按 Fisher Score 降序排序
        fisher_scores.sort(key=lambda x: x[1], reverse=True)

        # 选择 top_k 个特征
        top_features = [feature for feature, _ in fisher_scores[:top_k]]

        # 返回选择的特征及对应数据
        if target_col is not None:
            selected_data = data[top_features + [target_col]]
        else:
            selected_data = data[top_features]

        return selected_data, top_features

    def embedded_feature_selection(self, df, target_col, max_features=None, scoring='accuracy', cv=5):
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # Train Random Forest model
        rf = RandomForestClassifier(random_state=self.random_state)
        rf.fit(X, y)
        # Get feature importances
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        # Determine maximum number of features to consider
        max_features = max_features or len(feature_importances)
        # Perform cross-validation to find optimal number of features
        best_score = -float('inf')
        best_features = None
        for k in range(1, min(max_features + 1, len(feature_importances) + 1)):
            selected_features = feature_importances.head(k)['Feature'].tolist()
            X_selected = X[selected_features]
            try:
                scores = cross_val_score(rf, X_selected, y, scoring=scoring, cv=cv)
                mean_score = scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_features = selected_features
            except Exception as e:
                print(f"Error: Failed during cross-validation with {k} features: {e}")
                continue
        print("\n=== Embedded Feature Selection ===")
        print(feature_importances)
        # If no best features found, use top 10 or all features
        if best_features is None:
            print("Warning: Could not select optimal features. Using top 10 or all features.")
            best_features = feature_importances['Feature'].head(min(10, len(feature_importances))).tolist()
            best_score = 0
        print(f"Optimal number of features: {len(best_features)} Score: {best_score:.4f}")
        print(f"Selected features: {best_features}")
        return df[best_features + [target_col]], best_features

    def pca_feature_reduction(self, df, variance_threshold=0.95):
        """Reduces features using PCA to retain 95% of the variance."""
        pca = PCA(n_components=variance_threshold, random_state=self.random_state)
        reduced_data = pca.fit_transform(df)
        reduced_features = [f'PCA_{i + 1}' for i in range(reduced_data.shape[1])]
        df_reduced = pd.DataFrame(reduced_data, columns=reduced_features)
        print(f"PCA reduced features to {len(reduced_features)} components (95% variance retained).")
        return df_reduced, reduced_features

    def preprocess_features(self, X_raw, method='standard'):
        """
        Preprocess features using various scaling techniques.

        Parameters:
            X_raw (DataFrame): Raw feature matrix
            method (str): Preprocessing method ('none', 'standard', 'minmax', 'robust', 'pca')

        Returns:
            array-like: Processed feature matrix
        """
        if method == 'none':
            return X_raw.values
        elif method == 'standard':
            scaler = StandardScaler()
            return scaler.fit_transform(X_raw)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            return scaler.fit_transform(X_raw)
        elif method == 'robust':
            scaler = RobustScaler()
            return scaler.fit_transform(X_raw)
        elif method == 'normalize':
            scaler = Normalizer()
            return scaler.fit_transform(X_raw)
        elif method == 'pca':
            # First standardize
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X_raw)

            # Then apply PCA
            pca = PCA(n_components=0.9)  # Retain 95% variance
            X_pca = pca.fit_transform(X_std)
            print(f"PCA reduced features to {X_pca.shape[1]} components")
            return X_pca
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
class ManualClusterTuner:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def manual_gmm_clustering(self, X, n_components=4, covariance_type='full',
                              max_iter=100, tol=1e-3, init_params='kmeans',
                              reg_covar=1e-6):
        """
        Manually tune GMM clustering parameters.

        Parameters:
            X: feature matrix
            n_components: Number of mixture components
            covariance_type: Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
            max_iter: Maximum number of iterations
            tol: Convergence threshold
            init_params: Method for initialization ('kmeans', 'random')
            reg_covar: Regularization added to covariance
        """
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=tol,
            init_params=init_params,
            reg_covar=reg_covar,
            random_state=self.random_state
        )
        predicted_labels = gmm.fit_predict(X)
        return predicted_labels

    def manual_kmeans_clustering(self, X, n_clusters=4, init='k-means++',
                                 n_init=10, max_iter=100, tol=1e-4,
                                 algorithm='lloyd'):
        """
        Manually tune K-Means clustering parameters.

        Parameters:
            X: feature matrix
            n_clusters: Number of clusters
            init: Method for initialization ('k-means++', 'random')
            n_init: Number of initializations to perform
            max_iter: Maximum number of iterations
            tol: Convergence threshold
            algorithm: K-means algorithm to use ('lloyd', 'elkan')
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            algorithm=algorithm,
            random_state=self.random_state
        )
        predicted_labels = kmeans.fit_predict(X)
        return predicted_labels

    def manual_hierarchical_clustering(self, X, n_clusters=4, linkage='ward',
                                       metric='euclidean', connectivity=None):
        """
        Manually tune Hierarchical clustering parameters.

        Parameters:
            X: feature matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            metric: Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
            connectivity: Connectivity matrix (precomputed adjacency matrix)
        """
        # Note: 'ward' linkage only works with 'euclidean' metric
        if linkage == 'ward' and metric != 'euclidean':
            print("Warning: 'ward' linkage works only with 'euclidean' metric. Switching to 'euclidean'.")
            metric = 'euclidean'

        hc = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
            connectivity=connectivity
        )
        predicted_labels = hc.fit_predict(X)
        return predicted_labels
class ClusterOptimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def optimize_gmm(self, X, eval_data=None,
                     covariance_types=('full', 'tied', 'diag', 'spherical'),
                     max_iters=(1, 50, 100, 300, 500),
                     tols=(1e-3, 1e-6, 1e-9),
                     init_params=('kmeans', 'random'),
                     reg_covars=(5, 2, 1, 1e-1, 1e-2)):
        """
        Optimize GMM clustering (fixed 4 clusters) with intra/inter-cluster ratio.

        Parameters:
            X: Data to perform clustering on (should be pre-scaled if needed)
            eval_data: Optional data to evaluate clustering quality (if None, use X)
        """
        # Use data as-is without scaling
        X_data = X

        # Use eval_data as-is if provided, otherwise use X
        eval_data_for_ratio = eval_data if eval_data is not None else X_data

        best_results = []

        for covariance_type in covariance_types:
            for max_iter in max_iters:
                for tol in tols:
                    for init_param in init_params:
                        for reg_covar in reg_covars:
                            gmm = GaussianMixture(
                                n_components=4,
                                covariance_type=covariance_type,
                                max_iter=max_iter,
                                tol=tol,
                                init_params=init_param,
                                reg_covar=reg_covar,
                                random_state=self.random_state
                            )
                            labels = gmm.fit_predict(X_data)
                            ratio = ClusteringEvaluator.distance_ratio(eval_data_for_ratio, labels)

                            best_results.append({
                                'covariance_type': covariance_type,
                                'max_iter': max_iter,
                                'tol': tol,
                                'init_param': init_param,
                                'reg_covar': reg_covar,
                                'intra_inter_ratio': ratio,
                                'labels': labels
                            })

        return sorted(best_results, key=lambda x: x['intra_inter_ratio'])

    def optimize_kmeans(self, X, eval_data=None,
                        init_methods=('k-means++', 'random'),
                        n_init_values=(1, 2, 5, 10, 15),
                        max_iters=(50, 100, 300, 500),
                        tols=(1e-2, 1e-3, 1e-4),
                        algorithms=('lloyd', 'elkan')):
        """
        Optimize K-means clustering (fixed 4 clusters) with intra/inter-cluster ratio.

        Parameters:
            X: Data to perform clustering on (should be pre-scaled if needed)
            eval_data: Optional data to evaluate clustering quality (if None, use X)
        """
        # Use data as-is without scaling
        X_data = X

        # Use eval_data as-is if provided, otherwise use X
        eval_data_for_ratio = eval_data if eval_data is not None else X_data

        best_results = []

        for init in init_methods:
            for n_init in n_init_values:
                for max_iter in max_iters:
                    for tol in tols:
                        for algorithm in algorithms:
                            kmeans = KMeans(
                                n_clusters=4,
                                init=init,
                                n_init=n_init,
                                max_iter=max_iter,
                                tol=tol,
                                algorithm=algorithm,
                                random_state=self.random_state
                            )
                            labels = kmeans.fit_predict(X_data)
                            ratio = ClusteringEvaluator.distance_ratio(eval_data_for_ratio, labels)

                            best_results.append({
                                'init': init,
                                'n_init': n_init,
                                'max_iter': max_iter,
                                'tol': tol,
                                'algorithm': algorithm,
                                'intra_inter_ratio': ratio,
                                'labels': labels
                            })

        return sorted(best_results, key=lambda x: x['intra_inter_ratio'])

    def optimize_hierarchical(self, X, eval_data=None,
                              linkage_methods=('ward', 'complete', 'average', 'single'),
                              affinity_methods=('euclidean', 'manhattan', 'cosine')):
        """
        Optimize hierarchical clustering (fixed to 4 clusters) with intra/inter-cluster ratio.

        Parameters:
            X: Data to perform clustering on (should be pre-scaled if needed)
            eval_data: Optional data to evaluate clustering quality (if None, use X)
        """
        # Use data as-is without scaling
        X_data = X

        # Use eval_data as-is if provided, otherwise use X
        eval_data_for_ratio = eval_data if eval_data is not None else X_data

        best_results = []

        for linkage in linkage_methods:
            for metric in affinity_methods:
                if linkage == 'ward' and metric != 'euclidean':
                    continue

                hc = AgglomerativeClustering(
                    n_clusters=4,
                    linkage=linkage,
                    metric=metric
                )
                labels = hc.fit_predict(X_data)
                ratio = ClusteringEvaluator.distance_ratio(eval_data_for_ratio, labels)

                best_results.append({
                    'linkage': linkage,
                    'metric': metric,
                    'labels': labels,
                    'intra_inter_ratio': ratio
                })

        return sorted(best_results, key=lambda x: x['intra_inter_ratio'])
class ClusterOptimizer_report:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def optimize_gmm(self, X, true_labels, evaluation_metric, eval_data=None,
                     covariance_types=('full', 'tied', 'diag', 'spherical'),
                     max_iters=(1, 50, 100, 300, 500),
                     tols=(1e-3, 1e-6, 1e-9),
                     init_params=('kmeans', 'random'),
                     reg_covars=(5, 2, 1, 1e-1, 1e-2)):
        """
        Optimize GMM clustering with a custom evaluation metric.

        Parameters:
            X: Data to perform clustering on (should be pre-scaled if needed)
            true_labels: Ground truth labels for evaluation
            evaluation_metric: Function to evaluate clustering quality
            eval_data: Optional data to evaluate clustering quality (if None, use X)
        """
        X_data = X
        eval_data_for_metric = eval_data if eval_data is not None else X_data
        best_results = []

        for covariance_type in covariance_types:
            for max_iter in max_iters:
                for tol in tols:
                    for init_param in init_params:
                        for reg_covar in reg_covars:
                            gmm = GaussianMixture(
                                n_components=4,
                                covariance_type=covariance_type,
                                max_iter=max_iter,
                                tol=tol,
                                init_params=init_param,
                                reg_covar=reg_covar,
                                random_state=self.random_state
                            )
                            labels = gmm.fit_predict(X_data)
                            aligned_labels = ClusteringEvaluator.align_cluster_labels(true_labels, labels)
                            metric_value = evaluation_metric(true_labels, aligned_labels, eval_data_for_metric, labels)

                            best_results.append({
                                'covariance_type': covariance_type,
                                'max_iter': max_iter,
                                'tol': tol,
                                'init_param': init_param,
                                'reg_covar': reg_covar,
                                'metric_value': metric_value,
                                'labels': labels
                            })

        return sorted(best_results, key=lambda x: x['metric_value'], reverse=True)

    def optimize_kmeans(self, X, true_labels, evaluation_metric, eval_data=None,
                        init_methods=('k-means++', 'random'),
                        n_init_values=(1, 2, 5, 10, 15),
                        max_iters=(50, 100, 300, 500),
                        tols=(1e-2, 1e-3, 1e-4),
                        algorithms=('lloyd', 'elkan')):
        """
        Optimize K-means clustering with a custom evaluation metric.

        Parameters:
            X: Data to perform clustering on (should be pre-scaled if needed)
            true_labels: Ground truth labels for evaluation
            evaluation_metric: Function to evaluate clustering quality
            eval_data: Optional data to evaluate clustering quality (if None, use X)
        """
        X_data = X
        eval_data_for_metric = eval_data if eval_data is not None else X_data
        best_results = []

        for init in init_methods:
            for n_init in n_init_values:
                for max_iter in max_iters:
                    for tol in tols:
                        for algorithm in algorithms:
                            kmeans = KMeans(
                                n_clusters=4,
                                init=init,
                                n_init=n_init,
                                max_iter=max_iter,
                                tol=tol,
                                algorithm=algorithm,
                                random_state=self.random_state
                            )
                            labels = kmeans.fit_predict(X_data)
                            aligned_labels = ClusteringEvaluator.align_cluster_labels(true_labels, labels)
                            metric_value = evaluation_metric(true_labels, aligned_labels, eval_data_for_metric, labels)

                            best_results.append({
                                'init': init,
                                'n_init': n_init,
                                'max_iter': max_iter,
                                'tol': tol,
                                'algorithm': algorithm,
                                'metric_value': metric_value,
                                'labels': labels
                            })

        return sorted(best_results, key=lambda x: x['metric_value'], reverse=True)

    def optimize_hierarchical(self, X, true_labels, evaluation_metric, eval_data=None,
                              linkage_methods=('ward', 'complete', 'average', 'single'),
                              affinity_methods=('euclidean', 'manhattan', 'cosine')):
        """
        Optimize hierarchical clustering with a custom evaluation metric.

        Parameters:
            X: Data to perform clustering on (should be pre-scaled if needed)
            true_labels: Ground truth labels for evaluation
            evaluation_metric: Function to evaluate clustering quality
            eval_data: Optional data to evaluate clustering quality (if None, use X)
        """
        X_data = X
        eval_data_for_metric = eval_data if eval_data is not None else X_data
        best_results = []

        for linkage in linkage_methods:
            for metric in affinity_methods:
                if linkage == 'ward' and metric != 'euclidean':
                    continue

                hc = AgglomerativeClustering(
                    n_clusters=4,
                    linkage=linkage,
                    metric=metric
                )
                labels = hc.fit_predict(X_data)
                aligned_labels = ClusteringEvaluator.align_cluster_labels(true_labels, labels)
                metric_value = evaluation_metric(true_labels, aligned_labels, eval_data_for_metric, labels)

                best_results.append({
                    'linkage': linkage,
                    'metric': metric,
                    'metric_value': metric_value,
                    'labels': labels
                })

        return sorted(best_results, key=lambda x: x['metric_value'], reverse=True)

    def evaluate_ratio(self,true_labels, aligned_labels, eval_data, labels):
        return ClusteringEvaluator.distance_ratio(eval_data, labels)

    def evaluate_f1(self,true_labels, aligned_labels, eval_data, labels):
        accuracy, precision, recall, f1 = ClusteringEvaluator.calculate_metrics(true_labels, aligned_labels)
        return f1

    def evaluate_accuracy(self,true_labels, aligned_labels, eval_data, labels):
        accuracy, _, _, _ = ClusteringEvaluator.calculate_metrics(true_labels, aligned_labels)
        return accuracy

    def evaluate_ari(self,true_labels, aligned_labels, eval_data, labels):
        return ClusteringEvaluator.calculate_ari(true_labels, aligned_labels)
class ClusterParameterVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def _create_heatmap_facetgrid(df, row_var, col_var, index_var, columns_var, title, cbar_label='Intra/Inter Ratio'):
        """Helper function to avoid code duplication"""
        plt.close('all')  # Clear any existing figures

        # Create FacetGrid with unified settings
        g = sns.FacetGrid(df, col=col_var, row=row_var,
                          margin_titles=True, height=3, aspect=1.2,
                          despine=False)

        # Draw heatmaps
        def draw_heatmap(data, **kwargs):
            pivot_table = data.pivot_table(index=index_var, columns=columns_var, values='intra_inter_ratio')
            if not pivot_table.empty:  # Check if the pivot table is not empty
                sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu", cbar=False)

        g.map_dataframe(draw_heatmap)

        ## Set titles
        # g.set_titles(row_template=f'{row_var}={{row_name}}',
        #              col_template=f'{col_var}={{col_name}}')
        #
        # def set_dynamic_title(fig, title, y=1, width_ratio=0.8):
        #     fig_width = fig.get_size_inches()[0]  # Get the figure width in inches
        #     font_size = fig_width * 2.5* width_ratio  # Scale font size based on width
        #     fig.suptitle(title, y=y, fontsize=font_size)
        #
        # set_dynamic_title(g.fig, title, y=1)

        plt.tight_layout()
        plt.show()
        return g

    def visualize_gmm_parameters(self, results):
        df = pd.DataFrame(results)
        return self._create_heatmap_facetgrid(
            df,
            row_var='tol',
            col_var='max_iter',
            index_var='reg_covar',
            columns_var='covariance_type',
            title='Effect of GMM Parameters on Intra/Inter-Cluster Ratio'
        )

    def visualize_kmeans_parameters(self, results):
        df = pd.DataFrame(results)
        return self._create_heatmap_facetgrid(
            df,
            row_var='tol',
            col_var='max_iter',
            index_var='n_init',
            columns_var='algorithm',
            title='Effect of K-Means Parameters on Intra/Inter-Cluster Ratio'
        )

    def visualize_hierarchical_parameters(self, results):
        df = pd.DataFrame(results)
        return self._create_heatmap_facetgrid(
            df,
            # row_var='metric',
            # col_var='linkage',
            row_var=None,
            col_var=None,
            index_var='linkage',
            columns_var='metric',
            title='Effect of Hierarchical Parameters on Intra/Inter-Cluster Ratio'
        )
class ClusteringEvaluator:
    @staticmethod
    def distance_ratio(X, labels):
        """
        Evaluate the clustering result by calculating the ratio of intra-cluster distance
        to inter-cluster distance.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data.
        labels : array-like, shape (n_samples,)
            The cluster labels for each sample.

        Returns:
        float
            The ratio of intra-cluster distance to inter-cluster distance.
        """
        unique_labels = np.unique(labels)

        # Calculate intra-cluster distances
        intra_distances = []
        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                intra_distance = np.mean(pairwise_distances(cluster_points))
                intra_distances.append(intra_distance)

        # Calculate inter-cluster distances
        inter_distances = []
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                cluster_i = X[labels == unique_labels[i]]
                cluster_j = X[labels == unique_labels[j]]
                inter_distance = np.mean(pairwise_distances(cluster_i, cluster_j))
                inter_distances.append(inter_distance)

        # Calculate the average intra-cluster and inter-cluster distances
        avg_intra_distance = np.mean(intra_distances) if intra_distances else 0
        avg_inter_distance = np.mean(inter_distances) if inter_distances else 1  # Avoid division by zero

        # Calculate the ratio
        ratio = avg_intra_distance / avg_inter_distance if avg_inter_distance != 0 else float('inf')

        return ratio

    @staticmethod
    def calculate_ari(true_labels, predicted_labels):
        """
        计算 Adjusted Rand Index (ARI)
        :param true_labels: 真实标签
        :param predicted_labels: 聚类后的标签
        :return: ARI 分数
        """
        return adjusted_rand_score(true_labels, predicted_labels)

    @staticmethod
    def calculate_silhouette_score(X, predicted_labels):
        """
        计算轮廓系数 (Silhouette Score)
        :param X: 特征数据
        :param predicted_labels: 聚类后的标签
        :return: 轮廓系数分数
        """
        return silhouette_score(X, predicted_labels)

    @staticmethod
    def calculate_davies_bouldin_index(X, predicted_labels):
        return davies_bouldin_score(X, predicted_labels)
    @staticmethod
    def calculate_metrics(true_labels, predicted_labels):
        """Calculate accuracy, precision, recall, and F1 score."""
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        return accuracy, precision, recall, f1

    @staticmethod
    def align_cluster_labels(true_labels, predicted_labels):
        """
        将聚类标签与真实标签对齐，对齐方式是最大化F1分数

        参数:
            true_labels: 真实标签
            predicted_labels: 预测标签
        返回:
            对齐后的预测标签
        """
        # 获取唯一标签
        unique_pred_labels = np.unique(predicted_labels)
        unique_true_labels = np.sort(np.unique(true_labels))

        # 如果预测标签数量与真实标签数量不同
        if len(unique_pred_labels) != len(unique_true_labels):
            # 使用匈牙利算法进行初步映射
            cm = confusion_matrix(true_labels, predicted_labels)
            row_ind, col_ind = linear_sum_assignment(-cm)  # 取负值以进行最大化

            # 确保索引在范围内
            valid_col_ind = [i for i in col_ind if i < len(unique_pred_labels)]
            valid_row_ind = row_ind[:len(valid_col_ind)]

            # 创建映射
            mapping = {}
            for i in range(len(valid_col_ind)):
                old_label = unique_pred_labels[valid_col_ind[i]]
                new_label = unique_true_labels[valid_row_ind[i]]
                mapping[old_label] = new_label

            # 处理未映射的标签
            for label in unique_pred_labels:
                if label not in mapping:
                    unused_true_labels = [l for l in unique_true_labels if l not in mapping.values()]
                    mapping[label] = unused_true_labels[0] if unused_true_labels else -1
        else:
            # 尝试所有可能的映射，找到F1分数最高的
            best_f1 = 0
            best_mapping = {}

            # 生成真实标签的所有可能排列
            for perm in itertools.permutations(unique_true_labels):
                # 创建当前排列的映射
                current_mapping = {pred: true for pred, true in zip(unique_pred_labels, perm)}

                # 应用映射
                mapped_labels = np.array([current_mapping[label] for label in predicted_labels])

                # 计算F1分数
                current_f1 = f1_score(true_labels, mapped_labels, average='weighted')

                # 如果更好则更新
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_mapping = current_mapping

            mapping = best_mapping

        # 应用映射到预测标签
        aligned_labels = np.array([mapping[label] for label in predicted_labels])
        return aligned_labels

    @staticmethod
    def plot_confusion_matrix(true_labels, pred_labels, title="Confusion Matrix", show_plot=True):
        """
        绘制混淆矩阵并返回对齐后的预测标签

        参数:
            true_labels: 真实标签
            pred_labels: 预测标签
            title: 图表标题
            show_plot: 是否显示混淆矩阵图
        返回:
            aligned_labels: 对齐后的预测标签
        """
        # 获取对齐后的标签（确保聚类编号与真实类别一致）
        aligned_labels = ClusteringEvaluator.align_cluster_labels(true_labels, pred_labels)

        if show_plot:
            # 创建混淆矩阵
            conf_matrix = confusion_matrix(true_labels, aligned_labels)

            # 绘制混淆矩阵
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=sorted(np.unique(true_labels)),
                        yticklabels=sorted(np.unique(true_labels)))
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(title)
            plt.tight_layout()
            plt.show()

        return aligned_labels

class ManualClassifierTuner:
    """
    A class to manually tune and train various classifiers.
    """

    def __init__(self, random_state=None):
        """
        Initialize the classifier tuner.

        Parameters:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state

    def train_knn(self, features, labels, n_neighbors=11, weights='uniform',
                  algorithm='ball_tree', leaf_size=30, p=2, metric='chebyshev'):
        """
        Parameters:
            n_neighbors: int (default=5)
            weights: 'uniform', 'distance'
            algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'
            leaf_size: int (default=30)
            p: int (1=Manhattan, 2=Euclidean)
            metric: str (default='minkowski')
        """
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric
        )
        model.fit(features, labels)
        return model

    def train_decision_tree(self, features, labels, criterion='entropy',
                            splitter='best', max_depth=5, min_samples_split=10,
                            min_samples_leaf=4, max_features='sqrt'):
        """
        Parameters:
            criterion: 'gini', 'entropy'
            splitter: 'best', 'random'
            max_depth: int or None
            min_samples_split: int/float
            min_samples_leaf: int/float
            max_features: int/float/'sqrt'/'log2'/None
        """
        model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=self.random_state
        )
        model.fit(features, labels)
        return model

    def train_gnb(self, features, labels, priors=None, var_smoothing=0.6579):
        """
        Parameters:
            priors: array-like or None
            var_smoothing: float (default=1e-9)
        """
        model = GaussianNB(
            priors=priors,
            var_smoothing=var_smoothing
        )
        model.fit(features, labels)
        return model

    def train_random_forest(self, features, labels, n_estimators=50,
                            criterion='entropy', max_depth=5,
                            min_samples_split=2, min_samples_leaf=1,
                            max_features='sqrt', bootstrap=True):
        """
        Parameters:
            n_estimators: int (default=100)
            criterion: 'gini', 'entropy'
            max_depth: int or None
            min_samples_split: int/float
            min_samples_leaf: int/float
            max_features: int/float/'sqrt'/'log2'/None
            bootstrap: bool (default=True)
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=self.random_state,
        )
        model.fit(features, labels)
        return model

    def train_ensemble(self, features, labels, voting='soft', weights=None,
                       weight_strategy=None, knn_params=None, gnb_params=None,
                       dt_params=None, cv=5):


        if voting not in ['hard', 'soft']:
            raise ValueError("voting must be either 'hard' or 'soft'")

        if weight_strategy == 'confidence' and voting != 'soft':
            raise ValueError("confidence strategy requires soft voting")

        if dt_params is None:
            dt_params = {
                'criterion': 'entropy',
                'max_depth': 5,
                'max_features': 'sqrt',
                'min_samples_leaf': 4,
                'min_samples_split': 10,
                'splitter': 'best'
            }
        if knn_params is None:
            knn_params = {
                'n_neighbors': 11,
                'weights': 'uniform',
                'algorithm': 'ball_tree',
                'p': 1,
                'metric': 'chebyshev'
            }
        if gnb_params is None:
            gnb_params = {'priors': None, 'var_smoothing': 0.6579}

        # 初始化基础模型
        knn = KNeighborsClassifier(**knn_params)
        gnb = GaussianNB(**gnb_params)
        dt = DecisionTreeClassifier(**dt_params, random_state=self.random_state)

        estimators = [('knn', knn), ('gnb', gnb), ('dt', dt)]

        # 自动权重分配
        if weights is None and weight_strategy is not None:
            weights = self._compute_weights(
                estimators, features, labels,
                strategy=weight_strategy, cv=cv
            )

        # 创建并训练集成模型
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
        ensemble.fit(features, labels)

        return ensemble

    def _compute_weights(self, estimators, features, labels, strategy, cv=5):
        """
        计算模型权重的内部方法
        """
        if strategy == 'accuracy':
            return self._weights_by_metric(estimators, features, labels,
                                           'accuracy', cv)
        elif strategy == 'f1':
            return self._weights_by_metric(estimators, features, labels,
                                           'f1_weighted', cv)
        elif strategy == 'recall':
            return self._weights_by_metric(estimators, features, labels,
                                           'recall_weighted', cv)
        elif strategy == 'precision':
            return self._weights_by_metric(estimators, features, labels,
                                           'precision_weighted', cv)
        elif strategy == 'diversity':
            return self._weights_by_diversity(estimators, features, labels, cv)
        elif strategy == 'confidence':
            return self._weights_by_confidence(estimators, features, labels, cv)
        else:
            raise ValueError(f"Unknown weight strategy: {strategy}")

    def _weights_by_metric(self, estimators, features, labels, metric, cv):
        """
        基于评估指标计算权重
        """
        from sklearn.model_selection import cross_val_score
        scores = []
        for name, model in estimators:
            score = cross_val_score(model, features, labels, cv=cv,
                                    scoring=metric, n_jobs=-1)
            scores.append(np.mean(score))

        # 处理可能的负分(某些指标如log loss可能为负)
        scores = np.maximum(scores, 0)
        return np.array(scores) / np.sum(scores)

    def _weights_by_diversity(self, estimators, features, labels, cv):
        """
        基于模型多样性计算权重
        """

        # 获取各模型的交叉验证预测
        preds = []
        for name, model in estimators:
            pred = cross_val_predict(model, features, labels, cv=cv,
                                     method='predict', n_jobs=-1)
            preds.append(pred)

        # 计算每个模型与其他模型的平均差异度
        diversity = []
        for i in range(len(preds)):
            dists = []
            for j in range(len(preds)):
                if i != j:
                    dists.append(hamming(preds[i], preds[j]))
            diversity.append(np.mean(dists))

        return np.array(diversity) / np.sum(diversity)

    def _weights_by_confidence(self, estimators, features, labels, cv):
        """
        基于预测置信度计算权重(仅适用于soft voting)
        """
        from sklearn.model_selection import cross_val_predict

        confidences = []
        for name, model in estimators:
            proba = cross_val_predict(model, features, labels, cv=cv,
                                      method='predict_proba', n_jobs=-1)
            # 取预测概率的最大值作为置信度
            confidence = np.max(proba, axis=1).mean()
            confidences.append(confidence)

        return np.array(confidences) / np.sum(confidences)
class AutoClassifierTuner:
    """
    A class to automatically tune various classifiers using grid search.
    """

    def __init__(self, random_state=None, cv=5, n_jobs=-1, verbose=1,scoring='accuracy'):
        """
        Initialize the automatic classifier tuner.

        Parameters:
            random_state (int): Random seed for reproducibility
            cv (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs (-1 for all cores)
            verbose (int): Verbosity level (0-3)
        """
        self.random_state = random_state
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scoring = scoring

    def tune_knn(self, features, labels, param_grid=None):
        """
        Tune KNN classifier using grid search.

        Parameters:
            features (array-like): Feature matrix
            labels (array-like): Target labels
            param_grid (dict): Parameters to search, or None for default grid

        Returns:
            tuple: (Best model, Best parameters, Best cross-validation score)
        """


        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 13],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': [1, 2],  # 1=Manhattan, 2=Euclidean
                'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
            }

        # Create model
        knn = KNeighborsClassifier()

        # Set up grid search
        grid_search = GridSearchCV(
            knn,
            param_grid,
            cv=self.cv,
            # scoring='f1',
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        # Perform search
        grid_search.fit(features, labels)

        # Create and return best model
        best_model = KNeighborsClassifier(**grid_search.best_params_)
        best_model.fit(features, labels)

        return best_model, grid_search.best_params_, grid_search.best_score_

    def tune_decision_tree(self, features, labels, param_grid=None):
        """
        Tune Decision Tree classifier using grid search.

        Parameters:
            features (array-like): Feature matrix
            labels (array-like): Target labels
            param_grid (dict): Parameters to search, or None for default grid

        Returns:
            tuple: (Best model, Best parameters, Best cross-validation score)
        """


        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [ 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [None, 'sqrt', 'log2']
            }

        # Create model
        dt = DecisionTreeClassifier(random_state=self.random_state)

        # Set up grid search
        grid_search = GridSearchCV(
            dt,
            param_grid,
            cv=self.cv,
            # scoring='accuracy',
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        # Perform search
        grid_search.fit(features, labels)

        # Create and return best model
        best_params = grid_search.best_params_
        best_model = DecisionTreeClassifier(random_state=self.random_state, **best_params)
        best_model.fit(features, labels)

        return best_model, best_params, grid_search.best_score_

    def tune_gnb(self, features, labels, param_grid=None):
        """
        Tune Gaussian Naive Bayes classifier using grid search.

        Parameters:
            features (array-like): Feature matrix
            labels (array-like): Target labels
            param_grid (dict): Parameters to search, or None for default grid

        Returns:
            tuple: (Best model, Best parameters, Best cross-validation score)
        """


        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'var_smoothing': np.logspace(0, -9, num=100)
            }

        # Create model
        gnb = GaussianNB()

        # Set up grid search
        grid_search = GridSearchCV(
            gnb,
            param_grid,
            cv=self.cv,
            # scoring='accuracy',
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        # Perform search
        grid_search.fit(features, labels)

        # Create and return best model
        best_model = GaussianNB(**grid_search.best_params_)
        best_model.fit(features, labels)

        return best_model, grid_search.best_params_, grid_search.best_score_

    def tune_random_forest(self, features, labels, param_grid=None):
        """
        Tune Random Forest classifier using grid search.

        Parameters:
            features (array-like): Feature matrix
            labels (array-like): Target labels
            param_grid (dict): Parameters to search, or None for default grid

        Returns:
            tuple: (Best model, Best parameters, Best cross-validation score)
        """

        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'criterion': ['gini', 'entropy'],
                'max_depth': [4,5,6],
                'min_samples_split': [5,7],
                'min_samples_leaf': [6,8],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True]
            }

        # Create model
        rf = RandomForestClassifier(random_state=self.random_state)

        # Set up grid search
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=self.cv,
            # scoring='accuracy',
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        # Perform search
        grid_search.fit(features, labels)

        # Create and return best model
        best_params = grid_search.best_params_
        best_model = RandomForestClassifier(random_state=self.random_state, **best_params)
        best_model.fit(features, labels)

        return best_model, best_params, grid_search.best_score_

    def tune_ensemble(self, features, labels, param_grid=None):
        """
        Tune a VotingClassifier ensemble by first tuning each base model.

        Parameters:
            features (array-like): Feature matrix
            labels (array-like): Target labels
            param_grid (dict): Parameters to search for ensemble, or None for default grid

        Returns:
            tuple: (Best model, Best parameters, Best cross-validation score)
        """
        print("Tuning individual base models before ensemble...")

        # First tune each base model
        print("Tuning KNN...")
        best_knn, knn_params, knn_score = self.tune_knn(features, labels)
        print(f"Best KNN score: {knn_score:.4f}")

        print("Tuning GaussianNB...")
        best_gnb, gnb_params, gnb_score = self.tune_gnb(features, labels)
        print(f"Best GNB score: {gnb_score:.4f}")

        print("Tuning Decision Tree...")
        best_dt, dt_params, dt_score = self.tune_decision_tree(features, labels)
        print(f"Best DT score: {dt_score:.4f}")

        # Create base models with tuned parameters
        base_models = [
            ('knn', best_knn),
            ('gnb', best_gnb),
            ('dt', best_dt)
        ]

        # Default parameter grid if none provided
        if param_grid is None:
            weight_options = [1, 2, 3, 4, 5]
            weight_combinations = list(product(weight_options, repeat=len(base_models)))

            param_grid = {
                'voting': ['hard', 'soft'],
                'weights': [None] + [list(w) for w in weight_combinations]
            }

        print("Tuning ensemble with optimized base models...")

        # Create model
        ensemble = VotingClassifier(estimators=base_models)

        # Set up grid search
        grid_search = GridSearchCV(
            ensemble,
            param_grid,
            cv=self.cv,
            # scoring='accuracy',
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        # Perform search
        grid_search.fit(features, labels)

        # Create and return best model
        ensemble = VotingClassifier(
            estimators=base_models,
            **grid_search.best_params_
        )
        ensemble.fit(features, labels)

        # Return optimized ensemble model, parameters (including base model parameters), and score
        best_params = {
            'ensemble': grid_search.best_params_,
            'base_models': {
                'knn': knn_params,
                'gnb': gnb_params,
                'dt': dt_params
            }
        }

        return ensemble, best_params, grid_search.best_score_

    def tune_stacking(self, features, labels, cv_folds=5):
        """
        Implement a Stacking ensemble model with automatic hyperparameter tuning for base models.

        Parameters:
            features: Feature matrix
            labels: Target labels
            cv_folds: Number of cross-validation folds

        Returns:
            best_model: Best stacking model
            best_params: Best parameters
            best_score: Best cross-validation score
        """
        print("Tuning individual base models for stacking ensemble...")

        # First tune each base model
        print("\nTuning KNN...")
        best_knn, knn_params, knn_score = self.tune_knn(features, labels)
        print(f"Best KNN score: {knn_score:.4f}")

        print("\nTuning GaussianNB...")
        best_gnb, gnb_params, gnb_score = self.tune_gnb(features, labels)
        print(f"Best GNB score: {gnb_score:.4f}")

        print("\nTuning Decision Tree...")
        best_dt, dt_params, dt_score = self.tune_decision_tree(features, labels)
        print(f"Best DT score: {dt_score:.4f}")

        # Create base estimators with tuned models
        base_estimators = [
            ('knn', best_knn),
            ('gnb', best_gnb),
            ('dt', best_dt)
        ]

        print("\nConfiguring Stacking ensemble with tuned base models")
        print("Meta-learner: Logistic Regression")

        # Define meta-learner parameter grid
        param_grid = {
            'final_estimator__C': [0.01, 0.1],
            'final_estimator__solver': ['lbfgs', 'liblinear', 'saga'],
            'final_estimator__penalty': ['l2','l1'],
            'passthrough': [False]
        }

        # Create stacking model with LogisticRegression as meta-learner
        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=cv_folds
        )

        # Set up grid search
        print("Starting hyperparameter search for meta-learner...")
        grid_search = GridSearchCV(
            stacking,
            param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        # Perform search
        grid_search.fit(features, labels)
        print(f"Hyperparameter search complete. Best accuracy: {grid_search.best_score_:.4f}")

        # Create and train best model with the best meta-learner parameters
        meta_params = {k.replace("final_estimator__", ""): v for k, v in grid_search.best_params_.items()
                       if k.startswith("final_estimator__")}
        final_estimator = LogisticRegression(**meta_params, random_state=self.random_state)

        passthrough = grid_search.best_params_.get('passthrough', False)

        best_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=cv_folds,
            passthrough=passthrough
        )

        best_model.fit(features, labels)

        # Return the best model and all parameters
        best_params = {
            'meta_learner': meta_params,
            'passthrough': passthrough,
            'base_models': {
                'knn': knn_params,
                'gnb': gnb_params,
                'dt': dt_params
            }
        }

        return best_model, best_params, grid_search.best_score_

    def tune_customized_ensemble(self, data, target_column, feature_sets, preprocess_methods=None, param_grid=None):
        """
        Tune a VotingClassifier ensemble where each base model uses different feature sets and preprocessing methods.

        Parameters:
            data (DataFrame): Input data containing all features and target
            target_column (str): Name of the target column
            feature_sets (dict): Dictionary mapping model names to their feature lists
                e.g. {'knn': ['feat1', 'feat2'], 'gnb': ['feat1', 'feat3', 'feat4']}
            preprocess_methods (dict, optional): Dictionary mapping model names to preprocessing methods
                e.g. {'knn': 'standard', 'gnb': 'minmax', 'dt': 'robust'}
            param_grid (dict, optional): Parameters to search for ensemble, or None for default grid

        Returns:
            tuple: (Best ensemble model, Best parameters, Best cross-validation score)
        """
        # Extract target variable
        y = data[target_column]

        # Initialize feature engineering
        feature_engineer = FeatureEngineering(random_state=self.random_state)

        # Use default preprocessing methods if not provided
        if preprocess_methods is None:
            preprocess_methods = {
                'knn': 'standard',  # KNN works well with standardized data
                'gnb': 'minmax',  # GNB works well with normalized data
                'dt': 'none'  # Decision Trees don't require preprocessing
            }

        # Default feature sets if not provided in the feature_sets dict
        all_features = list(set().union(*[features for features in feature_sets.values() if features]))
        for model in ['knn', 'gnb', 'dt']:
            if model not in feature_sets or not feature_sets[model]:
                feature_sets[model] = all_features

        print("Tuning base models with customized feature sets and preprocessing methods...")

        # Initialize collection for fitted base models
        base_models = []
        model_details = {}

        # Train KNN with its feature set and preprocessing
        print(f"\nTuning KNN with {len(feature_sets['knn'])} features and {preprocess_methods['knn']} preprocessing...")
        X_knn = data[feature_sets['knn']]
        X_knn_processed = feature_engineer.preprocess_features(X_knn, method=preprocess_methods['knn'])
        best_knn, knn_params, knn_score = self.tune_knn(X_knn_processed, y)
        print(f"Best KNN score: {knn_score:.4f}")
        base_models.append(('knn', best_knn))
        model_details['knn'] = {
            'params': knn_params,
            'score': knn_score,
            'features': feature_sets['knn'],
            'preprocess': preprocess_methods['knn']
        }

        # Train GNB with its feature set and preprocessing
        print(
            f"\nTuning GaussianNB with {len(feature_sets['gnb'])} features and {preprocess_methods['gnb']} preprocessing...")
        X_gnb = data[feature_sets['gnb']]
        X_gnb_processed = feature_engineer.preprocess_features(X_gnb, method=preprocess_methods['gnb'])
        best_gnb, gnb_params, gnb_score = self.tune_gnb(X_gnb_processed, y)
        print(f"Best GNB score: {gnb_score:.4f}")
        base_models.append(('gnb', best_gnb))
        model_details['gnb'] = {
            'params': gnb_params,
            'score': gnb_score,
            'features': feature_sets['gnb'],
            'preprocess': preprocess_methods['gnb']
        }

        # Train Decision Tree with its feature set and preprocessing
        print(
            f"\nTuning Decision Tree with {len(feature_sets['dt'])} features and {preprocess_methods['dt']} preprocessing...")
        X_dt = data[feature_sets['dt']]
        X_dt_processed = feature_engineer.preprocess_features(X_dt, method=preprocess_methods['dt'])
        best_dt, dt_params, dt_score = self.tune_decision_tree(X_dt_processed, y)
        print(f"Best DT score: {dt_score:.4f}")
        base_models.append(('dt', best_dt))
        model_details['dt'] = {
            'params': dt_params,
            'score': dt_score,
            'features': feature_sets['dt'],
            'preprocess': preprocess_methods['dt']
        }

        # Default parameter grid if none provided
        if param_grid is None:
            weight_options = [1,2,3,4,5]
            weight_combinations = list(product(weight_options, repeat=len(base_models)))

            param_grid = {
                'voting': ['hard', 'soft'],
                'weights': [None] + [list(w) for w in weight_combinations]
            }

        print("\nTuning ensemble with optimized base models using different feature sets...")

        # For the ensemble training, we need to create merged feature data with all required features
        # and transform them according to each model's needs
        all_required_features = set()
        for model, details in model_details.items():
            all_required_features.update(details['features'])

        # Create a custom pipeline wrapper for the ensemble that handles different features for each model
        class CustomEnsemble(VotingClassifier):
            def __init__(self, estimators, voting='hard', weights=None, feature_sets=None, preprocess_methods=None):
                super().__init__(estimators=estimators, voting=voting, weights=weights)
                self.feature_sets = feature_sets
                self.preprocess_methods = preprocess_methods
                self.feature_engineer = FeatureEngineering()

            def fit(self, X, y, sample_weight=None):
                self.estimators_ = []
                self.named_estimators_ = {}

                # Fit each estimator on its own feature set with appropriate preprocessing
                for name, estimator in self.estimators:
                    if name in self.feature_sets and name in self.preprocess_methods:
                        features = self.feature_sets[name]
                        X_model = X[features]
                        X_processed = self.feature_engineer.preprocess_features(
                            X_model,
                            method=self.preprocess_methods[name]
                        )
                        estimator.fit(X_processed, y)
                        self.estimators_.append(estimator)
                        self.named_estimators_[name] = estimator

                return self

            def predict(self, X):
                predictions = []

                # Get predictions from each estimator using its own feature set
                for i, (name, _) in enumerate(self.estimators):
                    if name in self.feature_sets and name in self.preprocess_methods:
                        features = self.feature_sets[name]
                        X_model = X[features]
                        X_processed = self.feature_engineer.preprocess_features(
                            X_model,
                            method=self.preprocess_methods[name]
                        )
                        pred = self.estimators_[i].predict(X_processed)
                        predictions.append(pred)

                # Get the final prediction through voting
                if self.voting == 'hard':
                    return np.apply_along_axis(
                        lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                        axis=0,
                        arr=np.asarray(predictions).astype('int')
                    )
                else:  # soft voting
                    pred_proba = self.predict_proba(X)
                    return np.argmax(pred_proba, axis=1)

            def predict_proba(self, X):
                # Only available for 'soft' voting
                if self.voting != 'soft':
                    raise AttributeError("predict_proba is only available when voting='soft'")

                probas = []

                # Get probabilities from each estimator using its own feature set
                for i, (name, _) in enumerate(self.estimators):
                    if name in self.feature_sets and name in self.preprocess_methods:
                        features = self.feature_sets[name]
                        X_model = X[features]
                        X_processed = self.feature_engineer.preprocess_features(
                            X_model,
                            method=self.preprocess_methods[name]
                        )
                        proba = self.estimators_[i].predict_proba(X_processed)
                        probas.append(proba)

                # Apply weights if provided
                if self.weights is not None:
                    for i, w in enumerate(self.weights):
                        probas[i] = probas[i] * w

                # Average probabilities
                avg_proba = np.average(np.array(probas), axis=0)
                return avg_proba

        # Define a custom grid search for our specialized ensemble
        def custom_grid_search(param_grid, cv_splits=5):
            best_score = -1
            best_params = None
            y = data[target_column]

            # Create CV splits
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

            # Expand param grid
            param_list = list(ParameterGrid(param_grid))

            for params in param_list:
                # print(f"Evaluating: {params}")

                # Initialize model with current parameters
                ensemble = CustomEnsemble(
                    estimators=base_models,
                    voting=params.get('voting', 'soft'),
                    weights=params.get('weights', None),
                    feature_sets=feature_sets,
                    preprocess_methods=preprocess_methods
                )

                # Perform cross-validation
                scores = []
                for train_idx, test_idx in cv.split(data, y):
                    # Prepare train/test splits
                    X_train = data.iloc[train_idx]
                    y_train = y.iloc[train_idx]
                    X_test = data.iloc[test_idx]
                    y_test = y.iloc[test_idx]

                    # Fit model
                    ensemble.fit(X_train, y_train)

                    # Score model
                    y_pred = ensemble.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    scores.append(score)

                # Calculate mean score
                mean_score = np.mean(scores)
                # print(f"Mean CV score: {mean_score:.4f}")

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    print(f"New best: {best_score:.4f} with {best_params}")

            return best_params, best_score

        # Perform our custom grid search
        print("Starting grid search for ensemble parameters...")
        best_params, best_score = custom_grid_search(param_grid)

        # Create and fit final model with best parameters
        final_ensemble = CustomEnsemble(
            estimators=base_models,
            voting=best_params.get('voting', 'soft'),
            weights=best_params.get('weights', None),
            feature_sets=feature_sets,
            preprocess_methods=preprocess_methods
        )
        final_ensemble.fit(data, y)

        # Combine all parameters information
        all_params = {
            'ensemble': {
                'voting': best_params.get('voting'),
                'weights': best_params.get('weights'),
            },
            'models': model_details,
            'feature_sets': feature_sets,
            'preprocess_methods': preprocess_methods
        }

        return final_ensemble, all_params, best_score

    def tune_xgboost(self, features, labels, param_grid=None, cv=5):
        """
        Tune XGBoost classifier using grid search.
        """
        # 对标签进行编码确保从0开始
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)

        # 默认参数网格
        if param_grid is None:
            param_grid = {
                # 'n_estimators': [50, 100, 200],
                # 'max_depth': [3, 5, 7],
                # 'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100],
                'max_depth': [3],
                'learning_rate': [0.1],
                'subsample': [0.5],
                'colsample_bytree': [0.8],
                'gamma': [0.5],

                'reg_alpha': [1.0],  # 修正：使用列表而不是单个浮点数
                'reg_lambda': [10.0]  # 修正：使用列表而不是单个浮点数
            }

        # 初始化 XGBoost 分类器
        xgb = XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        # 设置网格搜索
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        # 执行网格搜索 - 使用编码后的标签
        grid_search.fit(features, encoded_labels)

        # 获取最佳模型
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # 使用最佳参数创建最终模型（仍然使用编码后的标签）
        best_model = XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
            **best_params
        )
        best_model.fit(features, encoded_labels)

        # 存储标签编码器供预测时使用
        best_model.label_encoder_ = le

        return best_model, best_params, best_score
class ClassifyingEvaluator:
    """
    A class to evaluate machine learning models using cross-validation.
    """

    @staticmethod
    def plot_learning_curve(estimator, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)):
        """
        Plot the learning curve with the estimator's name included in the title.

        Parameters:
            estimator: The machine learning model
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds
            scoring: Evaluation metric
            train_sizes: Proportions of the training set to use
        """
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=-1, random_state=42
        )

        # Calculate mean and standard deviation for training and validation scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Plot the learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color='r')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color='g')

        # Add title and labels
        estimator_name = type(estimator).__name__
        plt.title(f"Learning Curve for {estimator_name}")
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid()

        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, title="Confusion Matrix"):
        """
        Plot a confusion matrix.

        Parameters:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: List of class names (optional)
            normalize: Whether to normalize the confusion matrix
            title: Title of the confusion matrix plot
        """
        cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        # Plot the confusion matrix
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(title)
        plt.show()
    @staticmethod
    def calculate_accuracy(model, features, labels, cv=5):
        """
        Calculate the mean accuracy of a model using cross-validation.

        Parameters:
            model: The trained model to evaluate
            features (numpy.ndarray): Feature matrix
            labels (numpy.ndarray): Target labels
            cv (int): Number of cross-validation folds

        Returns:
            float: Mean accuracy across all folds
        """
        scores = cross_val_score(model, features, labels, cv=cv, scoring='accuracy')
        return np.mean(scores)

    @staticmethod
    def calculate_precision(model, features, labels, cv=5, average='weighted'):
        """
        Calculate the mean precision of a model using cross-validation.

        Parameters:
            model: The trained model to evaluate
            features (numpy.ndarray): Feature matrix
            labels (numpy.ndarray): Target labels
            cv (int): Number of cross-validation folds
            average (str): Averaging method ('micro', 'macro', 'weighted', 'binary')

        Returns:
            float: Mean precision across all folds
        """
        scores = cross_val_score(model, features, labels, cv=cv, scoring=f'precision_{average}')
        return np.mean(scores)

    @staticmethod
    def calculate_recall(model, features, labels, cv=5, average='weighted'):
        """
        Calculate the mean recall of a model using cross-validation.

        Parameters:
            model: The trained model to evaluate
            features (numpy.ndarray): Feature matrix
            labels (numpy.ndarray): Target labels
            cv (int): Number of cross-validation folds
            average (str): Averaging method ('micro', 'macro', 'weighted', 'binary')

        Returns:
            float: Mean recall across all folds
        """
        scores = cross_val_score(model, features, labels, cv=cv, scoring=f'recall_{average}')
        return np.mean(scores)

    @staticmethod
    def calculate_f1(model, features, labels, cv=5, average='weighted'):
        """
        Calculate the mean F1-score of a model using cross-validation.

        Parameters:
            model: The trained model to evaluate
            features (numpy.ndarray): Feature matrix
            labels (numpy.ndarray): Target labels
            cv (int): Number of cross-validation folds
            average (str): Averaging method ('micro', 'macro', 'weighted', 'binary')

        Returns:
            float: Mean F1-score across all folds
        """
        scores = cross_val_score(model, features, labels, cv=cv, scoring=f'f1_{average}')
        return np.mean(scores)

    @staticmethod
    def calculate_roc_auc(model, features, labels, cv=5, average='weighted'):
        """
        Calculate the mean ROC AUC of a model using cross-validation.
        Note: Only applicable for binary classification or with multi_class='ovr'.

        Parameters:
            model: The trained model to evaluate
            features (numpy.ndarray): Feature matrix
            labels (numpy.ndarray): Target labels
            cv (int): Number of cross-validation folds
            average (str): Averaging method ('micro', 'macro', 'weighted')

        Returns:
            float: Mean ROC AUC across all folds
        """
        try:
            scores = cross_val_score(model, features, labels, cv=cv, scoring=f'roc_auc_{average}')
            return np.mean(scores)
        except ValueError:
            # If ROC AUC is not applicable (e.g., for multiclass without probabilities)
            print("ROC AUC calculation not applicable for this model or dataset")
            return None

    @staticmethod
    def evaluate_model(model, features, labels, cv=5, average='weighted'):
        """
        Perform a comprehensive evaluation of a model using multiple metrics.

        Parameters:
            model: The trained model to evaluate
            features (numpy.ndarray): Feature matrix
            labels (numpy.ndarray): Target labels
            cv (int): Number of cross-validation folds
            average (str): Averaging method for metrics ('micro', 'macro', 'weighted')

        Returns:
            dict: Dictionary containing mean and standard deviation for each metric
        """

        scoring = {
            'accuracy': 'accuracy',
            'precision': f'precision_{average}',
            'recall': f'recall_{average}',
            'f1': f'f1_{average}'
        }

        # Try to add ROC AUC if applicable
        try:
            cv_results = cross_validate(
                model, features, labels, cv=cv,
                scoring={**scoring, 'roc_auc': f'roc_auc_{average}'}
            )
            has_roc_auc = True
        except ValueError:
            cv_results = cross_validate(model, features, labels, cv=cv, scoring=scoring)
            has_roc_auc = False

        # Calculate mean and std for each metric
        results = {}
        for metric in scoring.keys():
            results[metric] = {
                'mean': np.mean(cv_results[f'test_{metric}']),
                'std': np.std(cv_results[f'test_{metric}'])
            }

        if has_roc_auc:
            results['roc_auc'] = {
                'mean': np.mean(cv_results['test_roc_auc']),
                'std': np.std(cv_results['test_roc_auc'])
            }

        return results

    @staticmethod
    def evaluate_and_print(model, X_data, y_true, dataset_name):
        y_pred = model.predict(X_data)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"\n--- {model.__class__.__name__} on {dataset_name} Set ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred))

        return acc, prec, rec, f1

    @staticmethod
    def evaluate_xgb(model, X_data, y_true, dataset_name):
        # 使用存储的标签编码器进行预测，然后转换回原始标签
        y_pred_encoded = model.predict(X_data)
        y_pred = model.label_encoder_.inverse_transform(y_pred_encoded)

        # 计算指标
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"\n--- XGBoost on {dataset_name} Set ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred))

        return acc, prec, rec, f1