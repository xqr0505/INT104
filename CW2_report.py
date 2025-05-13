from CW_functions import *
import sys
import warnings


warnings.filterwarnings('ignore')
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"


def main():

    # matplotlib.use('TkAgg')  # 或者其他后端

    try:
        data = pd.read_excel('training_set.xlsx')
        print("Data loaded successfully.")
        print("\n=== DataFrame Headers ===")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print("\n=== First 5 rows ===")
        print(data.head())
    except FileNotFoundError:
        print("Error: 'training_set.xlsx' not found.")
        sys.exit(1)

    #====================#
    #   FeatureCreation  #
    #====================#

    original_features = ['Gender', 'Grade', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    # original_features = data.iloc[:, 1:].values
    question_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    target = 'Programme'

    # Initialize FeatureEngineering class
    feature_engineer = FeatureEngineering(random_state=42)

    # Create statistical features
    data, stat_features = feature_engineer.create_statistical_features(data, question_cols)
    print(f"Statistical Features Created: {stat_features}")

    # Create interaction features
    data, interact_features = feature_engineer.create_interaction_features(data, question_cols)
    print(f"Interaction Features Created: {interact_features}")


    #=======================#
    #   FeatureEngineering  #
    #=======================#

    # Filter correlated features
    new_features = stat_features + interact_features
    data, filtered_features = feature_engineer.filter_correlated_features(data, new_features, original_features)
    print(f"Filtered Features: {filtered_features}")

    # # Rank features
    # final_ranking = feature_engineer.rank_features(data, filtered_features, target)
    # print("Feature Ranking:")
    # print(final_ranking)
    #
    # # Rank features using Chi-square test
    # chi2_ranking = feature_engineer.chi2_feature_ranking(data, filtered_features, target)
    # print("\n=== Chi-Square Feature Ranking ===")
    # print(chi2_ranking)
    #
    # # Rank features using Mutual Information
    # mi_ranking = feature_engineer.mutual_info_feature_ranking(data, filtered_features, target)
    # print("\n=== Mutual Information Feature Ranking ===")
    # print(mi_ranking)
    #
    # # You can select top features based on each ranking method
    # top_n = 20  # Number of top features to select
    # top_chi2_features = chi2_ranking['Feature'].tolist()[:top_n]
    # top_mi_features = mi_ranking['Feature'].tolist()[:top_n]
    # top_final_ranking = final_ranking['Feature'].tolist()[:top_n]
    # print(f"Top {top_n} Chi-Square Features: {top_chi2_features}")
    # print(f"Top {top_n} Mutual Information Features: {top_mi_features}")
    # print(f"Top {top_n} Final Ranking Features: {top_final_ranking}")
    #
    #
    # # Perform RFECV with Logistic Regression
    # data_subset = data[filtered_features].copy()
    # data_subset[target] = data[target]  # Add target column for RFECV
    # selected_features, optimal_num_features = feature_engineer.perform_rfecv_with_logistic_regression(data_subset,target)
    # print(f"Selected Features with RFECV: {selected_features}")
    #
    #
    # # Perform similarity-based feature selection
    # data_with_target = data[filtered_features].copy()
    # data_with_target[target] = data[target]  # Add the target column
    # similarity_selected_data, similarity_features = feature_engineer.similarity_based_feature_selection(
    #     data_with_target, target, top_k=15)
    # print(f"Top 15 Features (Similarity-Based): {similarity_features}")
    #
    # # Perform Laplacian score-based feature selection
    # # Selects the top 5 features based on their Laplacian scores
    # laplacian_selected_data, laplacian_features = feature_engineer.laplacian_score_feature_selection(
    #     data[filtered_features], top_k=15)
    # print(f"Top 15 Features (Laplacian Score): {laplacian_features}")
    #
    # # Ensure the target column is included in the DataFrame
    # data_with_target = data[filtered_features + [target]].copy()
    #
    # # Perform embedded feature selection
    # selected_data, selected_features = feature_engineer.embedded_feature_selection(data_with_target, target,
    #                                                                                max_features=15)
    # print(f"Selected Features: {selected_features}")

    # ==================== #
    #   Feature Selection  #
    # ==================== #
    # Define feature sets
    feature_sets = {
        'original': ['Gender', 'Grade', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        'MI+CHI': ['Total_Score', 'Q1_div_Q2', 'Q1_div_Q4', 'Grade', 'Q2_plus_Q5', 'Q2_div_Q4', 'Q2', 'Q4_div_Q2',
                   'Q4', 'Q2_div_Q5', 'Q3_div_Q1', 'Q4_div_Q5', 'Q2_div_Q1', 'Min_Score', 'Q2_div_Q3'],
        'Similarity': ['Q1_div_Q4', 'Q3_div_Q1', 'Q1_div_Q2', 'Q2_div_Q4', 'Q1_absdiff_Q4', 'Q2_div_Q1',
                       'Q4_div_Q1', 'Q4_div_Q2', 'Q1_div_Q3', 'Q5_div_Q2', 'Q5_div_Q1', 'Q5_div_Q4',
                       'Q1_absdiff_Q2', 'Q2_div_Q3', 'Q5_div_Q3'],
        'Laplacian': ['Grade', 'Gender', 'Q5_div_Q3', 'Q4_div_Q3', 'Q2_div_Q3', 'Std_Dev', 'Min_Score',
                      'Q5', 'Q5_div_Q1', 'Q1_div_Q3', 'Q2_absdiff_Q5', 'Q5_div_Q4', 'Q1_absdiff_Q2',
                      'Q1_absdiff_Q4', 'Q2_absdiff_Q4'],
        'SFS GaussianNB': ['Std_Dev', 'Total_Score', 'Q1_plus_Q2', 'Q1_div_Q2', 'Q1_absdiff_Q3', 'Q3_div_Q1',
                           'Q1_absdiff_Q4', 'Q1_plus_Q5', 'Q1_absdiff_Q5', 'Q2_absdiff_Q4', 'Q2_plus_Q5',
                           'Q2_absdiff_Q5', 'Q2_div_Q5', 'Q3_absdiff_Q4', 'Q4_div_Q5', 'Gender', 'Grade', 'Q5'],
        'SFS KNN': ['Total_Score', 'Q1_plus_Q2', 'Q1_absdiff_Q2', 'Q1_div_Q2', 'Q2_div_Q1', 'Q1_absdiff_Q3',
                    'Q1_div_Q3', 'Q3_div_Q1', 'Q4_div_Q1', 'Q1_plus_Q5', 'Q1_absdiff_Q5', 'Q5_div_Q1', 'Q2_div_Q3',
                    'Q2_div_Q4',
                    'Q2_absdiff_Q5', 'Q5_div_Q4', 'Gender', 'Grade'],
        'SFS DecisionTree': ['Min_Score', 'Q1_absdiff_Q2', 'Q1_div_Q2', 'Q1_absdiff_Q3', 'Q3_div_Q1', 'Q1_div_Q4',
                             'Q4_div_Q1', 'Q1_absdiff_Q5', 'Q5_div_Q1', 'Q2_absdiff_Q4', 'Q5_div_Q2', 'Q5_div_Q3',
                             'Q4_div_Q5', 'Q5_div_Q4', 'Gender', 'Grade', 'Q4', 'Q5'],
        'SFS RandomForest': ['Total_Score', 'Q1_absdiff_Q3', 'Q3_div_Q1', 'Q4_div_Q1', 'Q1_plus_Q5', 'Q5_div_Q1',
                             'Q2_absdiff_Q4', 'Q2_div_Q4', 'Q4_div_Q2', 'Q2_plus_Q5', 'Q3_absdiff_Q4', 'Q5_div_Q3',
                             'Q4_div_Q5', 'Gender', 'Grade', 'Q2', 'Q4', 'Q5'],
        'Embedded RandomForest': ['Total_Score', 'Grade', 'Gender', 'Std_Dev', 'Q4_div_Q3', 'Q3_absdiff_Q4',
                                  'Q2_plus_Q5', 'Q4_div_Q2', 'Q1_div_Q3', 'Q4_div_Q5', 'Q1_absdiff_Q3', 'Q3',
                                  'Q1_plus_Q2', 'Q3_div_Q1', 'Q2_div_Q4'],
        'Fisher': ['Grade', 'Total_Score', 'Q2_plus_Q5', 'Q1_plus_Q2', 'Q2', 'Q4', 'Min_Score', 'Q1_plus_Q5', 'Q5',
                   'Q3',
                   'Q1_div_Q2', 'Gender', 'Q1_div_Q4', 'Q1', 'Q1_absdiff_Q4'],
        'test': ['Grade', 'Gender', 'Q5_div_Q3', 'Q4_div_Q3', 'Q2_div_Q3', 'Std_Dev', 'Min_Score',
                      'Q5',],
    }



    selected_feature_set = 'original'  # select feature sets
    selected_features = feature_sets[selected_feature_set]
    X_raw = data[selected_features]

    # ====================== #
    #   Data Preprocessing   #
    # ====================== #

    preprocess_method = 'standard'  # Options: 'none', 'standard', 'minmax', 'robust', 'pca'

    # Use the new method from FeatureEngineering class
    X_processed = feature_engineer.preprocess_features(X_raw, method=preprocess_method)

    y = data[target]

    # ============================ #
    #     Feature dimension        #
    # ============================ #
    feature_sets = {
        'MI+CHI': ['Grade','Total_Score', 'Q1_div_Q2', 'Q2_div_Q4', 'Q1_div_Q4', 'Q2', 'Q4_div_Q2', 'Q2_plus_Q5',
                   'Q4_div_Q5', 'Q5_div_Q4', 'Q2_div_Q1', 'Q3_div_Q1', 'Q1_plus_Q5', 'Q2_div_Q5', 'Q5',],
        'Fisher':  ['Grade','Total_Score', 'Q2_plus_Q5', 'Q1_plus_Q2', 'Q2', 'Q4', 'Min_Score', 'Q1_plus_Q5',
                    'Q5', 'Q3', 'Q1_div_Q2', 'Gender', 'Q1_div_Q4', 'Q1', 'Q1_absdiff_Q4',],

    }
    # Standardization processor
    scaler = StandardScaler()
    true_labels = data[target]  # True labels
    tuner = ManualClusterTuner(random_state=42)
    evaluator = ClusteringEvaluator()

    # 初始化结果存储
    results = {key: {'GMM': [], 'KMeans': [], 'Hierarchical': []} for key in feature_sets.keys()}
    #
    # 遍历每个特征集
    for feature_set_name, features in feature_sets.items():
        print(f"\n--- Testing feature set: {feature_set_name} ---")
        current_features = features.copy()

        while len(current_features) >= 1:
            # 提取和标准化特征
            X_raw = data[current_features]
            X_processed = StandardScaler().fit_transform(X_raw)
            true_labels = data[target]

            # 初始化结果字典
            result = {'Dimension': len(current_features)}

            # ==== GMM clustering ====
            gmm_labels = tuner.manual_gmm_clustering(X_processed)
            aligned_gmm_labels = evaluator.plot_confusion_matrix(true_labels, gmm_labels, show_plot=False)
            gmm_accuracy, _, _, gmm_f1 = evaluator.calculate_metrics(true_labels, aligned_gmm_labels)
            gmm_ari = evaluator.calculate_ari(true_labels, gmm_labels)
            gmm_ratio = evaluator.distance_ratio(X_processed, gmm_labels)
            # gmm_ratio = evaluator.distance_ratio(data[original_features], gmm_labels)
            results[feature_set_name]['GMM'].append((len(current_features), gmm_accuracy, gmm_f1, gmm_ari, gmm_ratio))

            # ==== K-Means clustering ====
            kmeans_labels = tuner.manual_kmeans_clustering(X_processed)
            aligned_kmeans_labels = evaluator.plot_confusion_matrix(true_labels, kmeans_labels, show_plot=False)
            kmeans_accuracy, _, _, kmeans_f1 = evaluator.calculate_metrics(true_labels, aligned_kmeans_labels)
            kmeans_ari = evaluator.calculate_ari(true_labels, kmeans_labels)
            kmeans_ratio = evaluator.distance_ratio(X_processed, kmeans_labels)
            # kmeans_ratio = evaluator.distance_ratio(data[original_features], kmeans_labels)
            results[feature_set_name]['KMeans'].append(
                (len(current_features), kmeans_accuracy, kmeans_f1, kmeans_ari, kmeans_ratio))

            # ==== Hierarchical clustering ====
            hc_labels = tuner.manual_hierarchical_clustering(X_processed)
            aligned_hc_labels = evaluator.plot_confusion_matrix(true_labels, hc_labels, show_plot=False)
            hc_accuracy, _, _, hc_f1 = evaluator.calculate_metrics(true_labels, aligned_hc_labels)
            hc_ari = evaluator.calculate_ari(true_labels, hc_labels)
            hc_ratio = evaluator.distance_ratio(X_processed, hc_labels)
            # hc_ratio = evaluator.distance_ratio(data[original_features], hc_labels)
            results[feature_set_name]['Hierarchical'].append(
                (len(current_features), hc_accuracy, hc_f1, hc_ari, hc_ratio))

            # 移除最后一个特征
            current_features.pop()

    # 绘制每个特征集的图
    for feature_set_name, clustering_results in results.items():
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Clustering Performance for {feature_set_name}', fontsize=16)

        metrics = ['Accuracy', 'F1', 'ARI', 'Ratio']
        for i, metric in enumerate(metrics):
            row, col = divmod(i, 2)
            ax = axs[row, col]

            for method, method_results in clustering_results.items():
                dimensions = [r[0] for r in method_results]
                values = [r[i + 1] for r in method_results]  # i+1 对应 Accuracy, F1, ARI, Ratio
                ax.plot(dimensions, values, marker='o', label=method, linewidth=2)

            ax.set_title(metric, fontsize=14)
            ax.set_xlabel('Feature Dimension', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            ax.set_xticks(dimensions)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    metrics = ['Accuracy', 'F1', 'ARI', 'Ratio']

    # 遍历每个聚类器
    for cluster_method in ['GMM', 'KMeans', 'Hierarchical']:
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'{cluster_method} Clustering Performance Across Feature Sets', fontsize=16)

        # 遍历每个指标
        for i, metric in enumerate(metrics):
            row, col = divmod(i, 2)
            ax = axs[row, col]

            # 用于检查是否有数据被绘制
            has_data = False

            # 遍历每个特征集
            for feature_set_name, clustering_results in results.items():
                if cluster_method in clustering_results and clustering_results[cluster_method]:
                    dimensions = [r[0] for r in clustering_results[cluster_method]]
                    metric_values = [r[i + 1] for r in clustering_results[cluster_method]]

                    if dimensions and metric_values:
                        ax.plot(dimensions, metric_values, marker='o', label=feature_set_name, linewidth=2)
                        has_data = True

                        # 找到全局最佳点
                        if metric == 'Ratio':  # Ratio 越低越好
                            best_value = min(metric_values)
                        else:  # 其他指标越高越好
                            best_value = max(metric_values)

                        best_dim = dimensions[metric_values.index(best_value)]

                        # 标注全局最佳点
                        ax.annotate(f'{best_value:.4f}',
                                    xy=(best_dim, best_value),
                                    xytext=(best_dim, best_value + (0.02 if metric != 'Ratio' else -0.02)),
                                    fontsize=8,
                                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

            # 设置子图标题和样式
            ax.set_title(metric, fontsize=14)
            ax.set_xlabel('Feature Dimension', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

            # 只有在有数据的情况下才设置图例和刻度
            if has_data:
                ax.legend(fontsize=10)
                if dimensions:
                    ax.set_xticks(dimensions)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 检查是否有任何数据被绘制，如果没有，则不显示空图
        show_plot = False
        for ax in axs.flatten():
            if len(ax.lines) > 0:
                show_plot = True
                break

        if show_plot:
            plt.show()
        else:
            plt.close(fig)  # 关闭空白图

    # 初始化结果存储
    results = {'GMM': [], 'KMeans': [], 'Hierarchical': []}


    y = data[target].values

    # 标准化特征
    scaler = StandardScaler()
    # 从数据集中提取数值型特征
    X_standardized = scaler.fit_transform(data[filtered_features])

    # 遍历PCA降维的维度数量
    for n_components in range(1, 16):
        # 应用PCA降维
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_standardized)

        # ==== GMM Clustering ====
        gmm_labels = tuner.manual_gmm_clustering(X_pca)
        gmm_accuracy, _, _, gmm_f1 = evaluator.calculate_metrics(y, gmm_labels)
        gmm_ari = evaluator.calculate_ari(y, gmm_labels)
        gmm_ratio = evaluator.distance_ratio(X_pca, gmm_labels)
        results['GMM'].append((n_components, gmm_accuracy, gmm_f1, gmm_ari, gmm_ratio))

        # ==== K-Means Clustering ====
        kmeans_labels = tuner.manual_kmeans_clustering(X_pca)
        kmeans_accuracy, _, _, kmeans_f1 = evaluator.calculate_metrics(y, kmeans_labels)
        kmeans_ari = evaluator.calculate_ari(y, kmeans_labels)
        kmeans_ratio = evaluator.distance_ratio(X_pca, kmeans_labels)
        results['KMeans'].append((n_components, kmeans_accuracy, kmeans_f1, kmeans_ari, kmeans_ratio))

        # ==== Hierarchical Clustering ====
        hc_labels = tuner.manual_hierarchical_clustering(X_pca)
        hc_accuracy, _, _, hc_f1 = evaluator.calculate_metrics(y, hc_labels)
        hc_ari = evaluator.calculate_ari(y, hc_labels)
        hc_ratio = evaluator.distance_ratio(X_pca, hc_labels)
        results['Hierarchical'].append((n_components, hc_accuracy, hc_f1, hc_ari, hc_ratio))

    # 绘制最终的图表
    # 绘制PCA降维的聚类性能图表
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Clustering Performance Across PCA Dimensions', fontsize=16)

    metrics = ['Accuracy', 'F1', 'ARI', 'Ratio']
    for i, metric in enumerate(metrics):
        row, col = divmod(i, 2)
        ax = axs[row, col]

        for method, method_results in results.items():
            dimensions = [r[0] for r in method_results]
            metric_values = [r[i + 1] for r in method_results]  # i+1 对应 Accuracy, F1, ARI, Ratio

            # 绘制每个聚类器的曲线
            ax.plot(dimensions, metric_values, marker='o', label=method, linewidth=2)

            # 找到全局最佳点
            if metric == 'Ratio':  # Ratio 越低越好
                best_value = min(metric_values)
            else:  # 其他指标越高越好
                best_value = max(metric_values)

            best_dim = dimensions[metric_values.index(best_value)]

            # 标注全局最佳点
            ax.annotate(f'{best_value:.4f}',
                        xy=(best_dim, best_value),
                        xytext=(best_dim, best_value + (0.02 if metric != 'Ratio' else -0.02)),
                        fontsize=8,
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1))

        # 设置子图标题和样式
        ax.set_title(metric, fontsize=14)
        ax.set_xlabel('PCA Dimensions', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        ax.set_xticks(dimensions)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\n=== Feature Dimension Optimization Analysis ===")

    # Initialize result storage
    dimension_results = []

    # Standardization processor
    scaler = StandardScaler()
    true_labels = data[target]  # True labels
    tuner = ManualClusterTuner(random_state=42)
    evaluator = ClusteringEvaluator()

    # Loop through each feature set
    for feature_set_name, features in feature_sets.items():
        print(f"\n--- Testing feature set: {feature_set_name} (Dimension: {len(features)}) ---")

        # Extract and standardize features
        X_raw = data[features]
        X_processed = scaler.fit_transform(X_raw)

        # Create result dictionary
        result = {
            'Feature Set': feature_set_name,
            'Dimension': len(features),
        }

        # ==== GMM clustering ====
        gmm_labels = tuner.manual_gmm_clustering(X_processed)
        # Don't show confusion matrix plot
        aligned_gmm_labels = evaluator.plot_confusion_matrix(true_labels, gmm_labels,
                                                             f"GMM Confusion Matrix ({feature_set_name})",
                                                             show_plot=False)

        # Calculate evaluation metrics
        gmm_accuracy, gmm_precision, gmm_recall, gmm_f1 = evaluator.calculate_metrics(true_labels,
                                                                                      aligned_gmm_labels)
        gmm_ari = evaluator.calculate_ari(true_labels, gmm_labels)
        gmm_ratio = evaluator.distance_ratio(X_processed, gmm_labels)

        # Store GMM results
        result.update({
            'GMM_Accuracy': gmm_accuracy,
            'GMM_F1': gmm_f1,
            'GMM_ARI': gmm_ari,
            'GMM_Ratio': gmm_ratio
        })

        # ==== K-Means clustering ====
        kmeans_labels = tuner.manual_kmeans_clustering(X_processed)
        aligned_kmeans_labels = evaluator.plot_confusion_matrix(true_labels, kmeans_labels,
                                                                f"K-Means Confusion Matrix ({feature_set_name})",
                                                                show_plot=False)

        # Calculate evaluation metrics
        kmeans_accuracy, kmeans_precision, kmeans_recall, kmeans_f1 = evaluator.calculate_metrics(true_labels,
                                                                                                  aligned_kmeans_labels)
        kmeans_ari = evaluator.calculate_ari(true_labels, kmeans_labels)
        kmeans_ratio = evaluator.distance_ratio(X_processed, kmeans_labels)

        # Store K-Means results
        result.update({
            'KMeans_Accuracy': kmeans_accuracy,
            'KMeans_F1': kmeans_f1,
            'KMeans_ARI': kmeans_ari,
            'KMeans_Ratio': kmeans_ratio
        })

        # ==== Hierarchical clustering ====
        hc_labels = tuner.manual_hierarchical_clustering(X_processed)
        aligned_hc_labels = evaluator.plot_confusion_matrix(true_labels, hc_labels,
                                                            f"Hierarchical Confusion Matrix ({feature_set_name})",
                                                            show_plot=False)

        # Calculate evaluation metrics
        hc_accuracy, hc_precision, hc_recall, hc_f1 = evaluator.calculate_metrics(true_labels,
                                                                                  aligned_hc_labels)
        hc_ari = evaluator.calculate_ari(true_labels, hc_labels)
        hc_ratio = evaluator.distance_ratio(X_processed, hc_labels)

        # Store hierarchical clustering results
        result.update({
            'HC_Accuracy': hc_accuracy,
            'HC_F1': hc_f1,
            'HC_ARI': hc_ari,
            'HC_Ratio': hc_ratio
        })

        # Print summary of current feature set results
        print(f"Dimension: {len(features)}")
        print(f"GMM - Accuracy: {gmm_accuracy:.4f}, F1: {gmm_f1:.4f}, ARI: {gmm_ari:.4f}, Ratio: {gmm_ratio:.4f}")
        print(
            f"K-Means - Accuracy: {kmeans_accuracy:.4f}, F1: {kmeans_f1:.4f}, ARI: {kmeans_ari:.4f}, Ratio: {kmeans_ratio:.4f}")
        print(f"Hierarchical - Accuracy: {hc_accuracy:.4f}, F1: {hc_f1:.4f}, ARI: {hc_ari:.4f}, Ratio: {hc_ratio:.4f}")

        # Add to results list
        dimension_results.append(result)

        # Close all plots
        plt.close('all')


    # =========================#
    #     OptimizeCluster      #
    # =========================#
    # 调用优化方法
    optimizer = ClusterOptimizer_report(random_state=42)

    # Define feature sets for each clustering algorithm and metric
    # 定义每个聚类算法和指标的特征集
    feature_sets = {
        'GMM': {
            'f1score': ['Grade', 'Total_Score', 'Q1_div_Q2'],
            'accuracy': ['Grade', 'Total_Score', 'Q2_plus_Q5', 'Q1_plus_Q2', 'Q2', 'Q4', 'Min_Score'],
            'ari': ['Grade', 'Total_Score']
        },
        'KMeans': {
            'f1score': ['Grade', 'Total_Score', 'Q1_div_Q2'],
            'accuracy': ['Grade', 'Total_Score', 'Q1_div_Q2'],
            'ari': ['Grade', 'Total_Score', 'Q1_div_Q2']
        },
        'Hierarchical': {
            'f1score': ['Grade', 'Total_Score', 'Q1_div_Q2',],
            'accuracy': ['Grade','Total_Score', 'Q2_plus_Q5', 'Q1_plus_Q2', 'Q2', 'Q4', 'Min_Score',],
            'ari': ['Grade', 'Total_Score']
        }
    }

    # 定义评估指标
    evaluation_metrics = {
        'f1score': optimizer.evaluate_f1,
        'accuracy': optimizer.evaluate_accuracy,
        'ari': optimizer.evaluate_ari
    }

    # 遍历每个聚类算法
    for algorithm, feature_config in feature_sets.items():
        print(f"\n=== Optimizing {algorithm} Clustering ===")

        # 遍历每个指标
        for metric_name, metric_function in evaluation_metrics.items():
            print(f"\n--- Optimizing for {metric_name.upper()} ---")

            # 获取 original 和对应的自定义特征集
            original_features = ['Gender', 'Grade', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
            custom_features = feature_config.get(metric_name)

            for feature_set_name, features in [('original', original_features), (metric_name, custom_features)]:
                print(f"\nFeature Set: {feature_set_name}")

                # 提取和标准化特征
                X_raw = data[features].values
                scaler = StandardScaler()
                X = scaler.fit_transform(X_raw)
                true_labels = data[target].values

                # 调用对应的优化方法
                if algorithm == 'GMM':
                    best_result = optimizer.optimize_gmm(X, true_labels, metric_function)
                elif algorithm == 'KMeans':
                    best_result = optimizer.optimize_kmeans(X, true_labels, metric_function)
                elif algorithm == 'Hierarchical':
                    best_result = optimizer.optimize_hierarchical(X, true_labels, metric_function)

                # 打印最佳参数和指标值
                best_params = best_result[0]
                print("Best Parameters:")
                print({key: value for key, value in best_params.items() if key != 'labels'})
                print(f"Best {metric_name.upper()} Value: {best_params['metric_value']:.4f}")

    #============================#
    #  confusion matrix(bestF1)  #
    #============================#

    features = ['Grade', 'Total_Score','Q1_div_Q2']
    X_raw = data[features].values
    true_labels = data[target].values

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    #使用指定的GMM参数进行聚类
    gmm = GaussianMixture(
        n_components=len(np.unique(true_labels)),  # 聚类数等于真实标签的类别数
        covariance_type='spherical',
        max_iter=50,
        tol=1e-6,
        init_params='kmeans',
        reg_covar=0.01,
        random_state=42
    )

    gmm.fit(X)
    predicted_labels = gmm.predict(X)

    # 绘制混淆矩阵
    evaluator = ClusteringEvaluator()
    evaluator.plot_confusion_matrix(true_labels, predicted_labels, title="GMM Confusion Matrix")
    aligned_labels = evaluator.align_cluster_labels(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, aligned_labels)
    print(conf_matrix)

if __name__ == "__main__":
    main()


