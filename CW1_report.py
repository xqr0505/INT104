import sys

from sklearn.manifold import TSNE

from CW_functions import *


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

    # #====================#
    # #     observation    #
    # #====================#
    # # 原始特征
    # original_features = ['Grade', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    # question_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    # target = 'Programme'
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(data[original_features])
    #
    # # 设置调色板
    # palette = sns.color_palette("Set2", n_colors=data[target].nunique())
    #
    # # 将数据转换为长格式以便绘制单个 boxplot
    # data_long = data.melt(id_vars=[target], value_vars=original_features,
    #                       var_name='Feature', value_name='Value')
    #
    # # 绘制单个 boxplot
    # plt.figure(figsize=(12, 8))
    # sns.boxplot(x='Feature', y='Value', hue=target, data=data_long, palette=palette)
    # plt.title("Boxplot of All Features by Programme")
    # plt.xticks(rotation=45)
    # plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.show()
    #
    # # PCA 可视化
    # pca = PCA(n_components=2, random_state=42)
    # pca_result = pca.fit_transform(X_scaled)
    # plt.figure(figsize=(8, 6))
    # for i, label in enumerate(data[target].unique()):
    #     plt.scatter(
    #         pca_result[data[target] == label, 0],
    #         pca_result[data[target] == label, 1],
    #         label=label,
    #         color=palette[i],
    #         alpha=0.6
    #     )
    # plt.title("PCA Visualization")
    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    # plt.legend(title=target)
    # plt.show()
    #
    # # t-SNE 可视化
    # tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    # tsne_result = tsne.fit_transform(X_scaled)
    # plt.figure(figsize=(8, 6))
    # for i, label in enumerate(data[target].unique()):
    #     plt.scatter(
    #         tsne_result[data[target] == label, 0],
    #         tsne_result[data[target] == label, 1],
    #         label=label,
    #         color=palette[i],
    #         alpha=0.6
    #     )
    # plt.title("t-SNE Visualization")
    # plt.xlabel("t-SNE Component 1")
    # plt.ylabel("t-SNE Component 2")
    # plt.legend(title=target)
    # plt.show()
    #
    # # 计算原始数据的统计量
    # original_data = data[original_features].select_dtypes(include=np.number)
    # original_value_range = original_data.max() - original_data.min()
    # original_mean = original_data.mean()
    # original_data = data[original_features].select_dtypes(include=np.number)
    # sample_norm = np.linalg.norm(original_data, axis=1)
    #
    # print("\n=== Original Data ===")
    # print("Value Range:")
    # print(original_value_range.round(2))
    # print("Mean:")
    # print(original_mean.round(2))
    # print("Sample Norm Range (Original Data):")
    # print(f"Min: {sample_norm.min():.2f}, Max: {sample_norm.max():.2f}")
    #
    # # 归一化方法
    # scalers = {
    #     "Z-score": StandardScaler(),
    #     "Min-Max": MinMaxScaler(),
    #     "L2 Normalize": Normalizer(norm='l2')
    # }
    #
    # # 对每种归一化方法进行处理并计算统计量
    # for name, scaler in scalers.items():
    #     normalized_data = scaler.fit_transform(data[original_features].select_dtypes(include=np.number))
    #     normalized_df = pd.DataFrame(normalized_data, columns=original_features)
    #
    #     # 计算 Value Range 和 Mean
    #     value_range = normalized_df.max() - normalized_df.min()
    #     mean_post_norm = normalized_df.mean()
    #
    #     # 计算 Sample Norm 范围
    #     sample_norm = np.linalg.norm(normalized_data, axis=1)
    #     sample_norm_range = (sample_norm.min(), sample_norm.max())
    #
    #     print(f"\n=== {name} Normalization ===")
    #     print("Value Range:")
    #     print(value_range.round(2))
    #     print("Mean (Post-Normalization):")
    #     print(mean_post_norm.round(2))
    #     print("Sample Norm Range:")
    #     print(f"Min: {sample_norm_range[0]:.2f}, Max: {sample_norm_range[1]:.2f}")

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

    # =======================#
    #   FeatureEngineering  #
    # =======================#

    # Filter correlated features
    new_features = stat_features + interact_features
    data, filtered_features = feature_engineer.filter_correlated_features(data, new_features, original_features)
    print(f"Filtered Features: {filtered_features}")

    data_with_target = data[filtered_features].copy()
    data_with_target[target] = data[target]  # Add the target column
    if 'Grade' in filtered_features:
        filtered_features.remove('Grade')

    # Rank features
    final_ranking = feature_engineer.rank_features(data, filtered_features, target)
    print("Feature Ranking:")
    print(final_ranking)

    # You can select top features based on each ranking method
    top_n = 20  # Number of top features to select
    top_final_ranking = final_ranking['Feature'].tolist()[:top_n]
    print(f"Top {top_n} Final Ranking Features: {top_final_ranking}")


    # # Perform similarity-based feature selection
    # similarity_selected_data, similarity_features = feature_engineer.similarity_based_feature_selection(
    #     data_with_target, target, top_k=15)
    # print(f"Top 15 Features (Similarity-Based): {similarity_features}")
    #
    # # Perform Laplacian score-based feature selection
    # laplacian_selected_data, laplacian_features = feature_engineer.laplacian_score_feature_selection(
    #     data[filtered_features], top_k=15)
    # print(f"Top 15 Features (Laplacian Score): {laplacian_features}")

    # Perform Fisher score-based feature selection
    fisher_selected_data, fisher_features = feature_engineer.fisher_score_feature_selection(
        data_with_target, target, top_k=top_n)
    print(f"Top 15 Features (Fisher Score): {fisher_features}")

    # #  Use Sequential Forward Selection with multiple models
    # data_subset = data[filtered_features].copy()
    # X = data_subset
    # y = data[target]
    #
    # # Perform Sequential Forward Selection
    # sfs_results = feature_engineer.perform_sequential_forward_selection(X, y)
    # print("\n=== Sequential Forward Selection Results ===")
    # for model_name, selected_feats in sfs_results.items():
    #     print(f"{model_name}: {selected_feats}")
    #     print(f"Number of features selected: {len(selected_feats)}")
    # # Ensure the target column is included in the DataFrame


    # Perform embedded feature selection
    selected_data, selected_features = feature_engineer.embedded_feature_selection(data_with_target, target)


    # ==================== #
    #   Feature Selection  #
    # ==================== #
    # Define feature sets
    feature_sets = {
        'original(7 features)': ['Gender', 'Grade', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        'MI+CHI(top 7)': ['Total_Score', 'Q1_div_Q2', 'Q1_div_Q4', 'Grade', 'Q2_plus_Q5', 'Q2_div_Q4', 'Q2'],
        'Fisher(top7)': ['Grade', 'Total_Score', 'Q2_plus_Q5', 'Q1_plus_Q2', 'Q2', 'Q4', 'Min_Score',],
        'Embedded RandomForest(top 7)': ['Total_Score', 'Grade', 'Gender', 'Std_Dev', 'Q4_div_Q3', 'Q3_absdiff_Q4',
                                         'Q2_plus_Q5'],
        'MI+CHI(top 18)': ['Total_Score', 'Q1_div_Q2', 'Q1_div_Q4', 'Grade', 'Q2_plus_Q5', 'Q2_div_Q4', 'Q2', 'Q4_div_Q2',
                   'Q4', 'Q2_div_Q5', 'Q3_div_Q1', 'Q4_div_Q5', 'Q2_div_Q1', 'Min_Score', 'Q2_div_Q3', 'Q5',
                   'Q1_absdiff_Q4', 'Q4_div_Q1'],
        'Fisher(top18)': ['Grade', 'Total_Score', 'Q2_plus_Q5', 'Q1_plus_Q2', 'Q2', 'Q4', 'Min_Score', 'Q1_plus_Q5', 'Q5',
                          'Q3', 'Q1_div_Q2', 'Gender', 'Q1_div_Q4', 'Q1', 'Q1_absdiff_Q4', 'Q1_absdiff_Q2', 'Q2_absdiff_Q5',
                          'Q4_div_Q2'],
        'SFS GaussianNB(18 features)': ['Std_Dev', 'Total_Score', 'Q1_plus_Q2', 'Q1_div_Q2', 'Q1_absdiff_Q3', 'Q3_div_Q1',
                       'Q1_absdiff_Q4','Q1_plus_Q5', 'Q1_absdiff_Q5', 'Q2_absdiff_Q4', 'Q2_plus_Q5',
                        'Q2_absdiff_Q5', 'Q2_div_Q5','Q3_absdiff_Q4', 'Q4_div_Q5', 'Gender', 'Grade', 'Q5'],
        'SFS KNN(18 features)': ['Total_Score', 'Q1_plus_Q2', 'Q1_absdiff_Q2', 'Q1_div_Q2', 'Q2_div_Q1', 'Q1_absdiff_Q3',
                    'Q1_div_Q3','Q3_div_Q1', 'Q4_div_Q1', 'Q1_plus_Q5', 'Q1_absdiff_Q5', 'Q5_div_Q1', 'Q2_div_Q3', 'Q2_div_Q4',
                    'Q2_absdiff_Q5', 'Q5_div_Q4', 'Gender', 'Grade'],
        'SFS DecisionTree(18 features)': ['Min_Score', 'Q1_absdiff_Q2', 'Q1_div_Q2', 'Q1_absdiff_Q3', 'Q3_div_Q1', 'Q1_div_Q4',
                             'Q4_div_Q1', 'Q1_absdiff_Q5', 'Q5_div_Q1', 'Q2_absdiff_Q4', 'Q5_div_Q2', 'Q5_div_Q3',
                             'Q4_div_Q5', 'Q5_div_Q4', 'Gender', 'Grade', 'Q4', 'Q5'],
        'SFS RandomForest(18 features)': ['Total_Score', 'Q1_absdiff_Q3', 'Q3_div_Q1', 'Q4_div_Q1', 'Q1_plus_Q5', 'Q5_div_Q1',
                             'Q2_absdiff_Q4', 'Q2_div_Q4', 'Q4_div_Q2', 'Q2_plus_Q5', 'Q3_absdiff_Q4', 'Q5_div_Q3',
                             'Q4_div_Q5', 'Gender', 'Grade', 'Q2', 'Q4', 'Q5'],
        'Embedded RandomForest': ['Total_Score', 'Gender']
    }


    selected_feature_set = 'Fisher(top7)'  # select feature sets
    selected_features = feature_sets[selected_feature_set]
    X_raw = data[selected_features]


    # ====================== #
    #   Data Preprocessing   #
    # ====================== #

    preprocess_method = 'standard'  # Options: 'none', 'standard', 'minmax', 'robust', 'pca'

    # Use the new method from FeatureEngineering class
    X_processed = feature_engineer.preprocess_features(X_raw, method=preprocess_method)

    y = data[target]


    # ============================#
    #       t-sne visualize       #
    # ============================#

    # t-SNE 可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(X_processed)
    palette = sns.color_palette("Set2", n_colors=data[target].nunique())
    # 绘制 t-SNE 可视化
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(y.unique()):
        plt.scatter(
            tsne_result[y == label, 0],
            tsne_result[y == label, 1],
            label=f"Class {label}",
            color=palette[i],
            alpha=0.6
        )
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Classes")
    plt.show()


    # ============================#
    #       KMeans Clustering     #
    # ============================#
    # 定义 KMeans 聚类器
    n_clusters = len(y.unique())
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_processed)

    # 计算聚类结果的距离比率
    kmeans_ratio = ClusteringEvaluator.distance_ratio(X_processed, kmeans_labels)
    print(f"K-means clustering distance ratio: {kmeans_ratio:.4f}")

    # 将聚类标签添加到原始数据中
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = kmeans_labels

    # 打印 cluster 1 的样本原始数据
    cluster1_samples = data_with_clusters[data_with_clusters['Cluster'] == 1]
    print(f"\n共有 {len(cluster1_samples)} 个样本被分到 Cluster 1")
    print("\n=== Cluster 1 样本的原始数据 ===")

    # 只显示选定的特征
    print(cluster1_samples[selected_features].head(10))  # 显示前10条数据

    # =====================================#
    #       t-SNE Visualization (KMeans)   #
    # =====================================#
    # t-SNE 可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(X_processed)
    palette = sns.color_palette("Set2", n_colors=n_clusters)

    # 绘制 t-SNE 可视化
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(
            tsne_result[kmeans_labels == i, 0],
            tsne_result[kmeans_labels == i, 1],
            label=f"Cluster {i}",
            color=palette[i],
            alpha=0.6
        )
    plt.title("t-SNE Visualization (KMeans Clusters)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Clusters")
    plt.show()

    # ============================================ #
    #  Distance Ratio for different feature sets   #
    # ============================================ #
    X_array = X_processed  # 直接使用 numpy 数组
    labels = data[target].values  # 获取组别标签
    distance_ratio = ClusteringEvaluator.distance_ratio(X_array, labels)

    print(f"Distance Ratio (Intra-cluster / Inter-cluster): {distance_ratio:.4f}")

    # 定义预处理方法
    preprocessing_methods = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'pca': PCA(n_components=0.9, random_state=42)
    }

    # 初始化结果存储
    results = []

    # 遍历每个特征集
    for feature_set_name, features in feature_sets.items():
        row = {'Feature Set': feature_set_name}
        X_raw = data[features].values  # 获取特征数据

        # 遍历每种预处理方法
        for method_name, method in preprocessing_methods.items():
            if method_name == 'pca':  # PCA 特殊处理
                X_processed = StandardScaler().fit_transform(X_raw)  # PCA 需要先标准化
                X_processed = method.fit_transform(X_processed)
            else:
                X_processed = method.fit_transform(X_raw)

            # 计算 distance_ratio
            labels = data[target].values
            distance_ratio = ClusteringEvaluator.distance_ratio(X_processed, labels)
            row[method_name] = round(distance_ratio, 2)  # 保留两位小数

        results.append(row)

    # 转换为 DataFrame
    results_df = pd.DataFrame(results)

    # 打印表格
    print("\n=== Distance Ratio Table ===")
    print(results_df)

    # 找到最小值及其对应的特征集和预处理方法
    min_value = results_df.iloc[:, 1:].min().min()  # 找到最小值
    min_method = results_df.iloc[:, 1:].idxmin(axis=1)[results_df.iloc[:, 1:].min(axis=1).idxmin()]  # 最小值对应的列
    min_feature_set = results_df.loc[results_df.iloc[:, 1:].min(axis=1).idxmin(), 'Feature Set']

    print(f"\nMinimum Distance Ratio: {min_value:.2f}")
    print(f"Feature Set: {min_feature_set}, Preprocessing Method: {min_method}")

    # Plot line chart to visualize distance ratio across different preprocessing methods
    plt.figure(figsize=(12, 6))

    # Prepare X-axis labels, extract numeric part from feature set names for sorting
    # Assuming feature set names have format like 'MI+CHI15', extract the numeric part
    feature_numbers = []
    for feature_set in results_df['Feature Set']:
        try:
            # Extract numeric part from feature set name
            num = int(''.join(filter(str.isdigit, feature_set)))
            feature_numbers.append(num)
        except:
            feature_numbers.append(0)  # Set to 0 if no number can be extracted

    # Sort by feature count
    results_df['Feature_Number'] = feature_numbers
    sorted_df = results_df.sort_values('Feature_Number')

    # Plot three lines, each representing a preprocessing method
    for method in ['standard', 'minmax', 'pca']:
        plt.plot(
            sorted_df['Feature Set'],
            sorted_df[method],
            marker='o',
            linewidth=2,
            label=f'{method.capitalize()}'
        )

    # Mark the best point
    best_idx = sorted_df[sorted_df['Feature Set'] == min_feature_set].index[0]
    plt.scatter(
        min_feature_set,
        min_value,
        s=150,
        color='red',
        marker='*',
        label=f'Best: {min_feature_set} + {min_method}'
    )

    # Chart style settings
    plt.title('Distance Ratio Comparison')
    plt.xlabel('Feature Sets')
    plt.ylabel('Distance Ratio (Lower is Better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.xticks(rotation=45)  # Rotate X-axis labels to prevent overlap

    # Add horizontal reference line
    plt.axhline(y=min_value, color='r', linestyle='--', alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # ====================== #
    #    reversed features   #
    # ====================== #
    feature_sets = {
        'MI+CHI': ['Total_Score', 'Q1_div_Q2', 'Q1_div_Q4', 'Q2_plus_Q5', 'Q2_div_Q4', 'Q2', 'Q4_div_Q2',
                   'Q4', 'Q2_div_Q5', 'Q3_div_Q1', 'Q4_div_Q5', 'Q2_div_Q1', 'Min_Score', 'Q2_div_Q3', 'Q5',
                   'Q1_absdiff_Q4', 'Q4_div_Q1'],
        'Fisher': ['Total_Score', 'Q2_plus_Q5', 'Q1_plus_Q2', 'Q2', 'Q4', 'Min_Score', 'Q1_plus_Q5',
                   'Q5', 'Q3', 'Q1_div_Q2', 'Gender', 'Q1_div_Q4', 'Q1', 'Q1_absdiff_Q4', 'Q1_absdiff_Q2',
                   'Q2_absdiff_Q5', 'Q4_div_Q2'],
        'Embedded RandomForest': ['Total_Score', 'Gender', 'Std_Dev', 'Q4_div_Q3', 'Q3_absdiff_Q4',
                                  'Q2_plus_Q5', 'Q4_div_Q2', 'Q1_div_Q3', 'Q4_div_Q5', 'Q1_absdiff_Q3', 'Q3',
                                  'Q1_plus_Q2', 'Q3_div_Q1', 'Q2_div_Q4', 'Q2_div_Q3', 'Q4_div_Q1',
                                  'Q2_absdiff_Q4']
    }

    # 逆序处理
    reversed_feature_sets = {key: list(reversed(value)) for key, value in feature_sets.items()}

    # 输出结果
    print(reversed_feature_sets)
    # ==================== #
    #   Feature dimension  #
    # ==================== #
    # Define the feature sets
    feature_sets = {
        'MI+CHI': ['Total_Score', 'Q1_div_Q2', 'Q1_div_Q4', 'Q2_plus_Q5', 'Q2_div_Q4', 'Q2','Q4_div_Q2',
                   'Q4', 'Q2_div_Q5', 'Q3_div_Q1', 'Q4_div_Q5', 'Q2_div_Q1', 'Min_Score', 'Q2_div_Q3', 'Q5',
                   'Q1_absdiff_Q4', 'Q4_div_Q1'],
        'Fisher': ['Total_Score', 'Q2_plus_Q5', 'Q1_plus_Q2', 'Q2', 'Q4', 'Min_Score', 'Q1_plus_Q5',
                   'Q5','Q3', 'Q1_div_Q2', 'Gender', 'Q1_div_Q4', 'Q1', 'Q1_absdiff_Q4', 'Q1_absdiff_Q2',
                   'Q2_absdiff_Q5','Q4_div_Q2'],
        'Embedded RandomForest': ['Total_Score',  'Gender', 'Std_Dev', 'Q4_div_Q3', 'Q3_absdiff_Q4',
                                  'Q2_plus_Q5', 'Q4_div_Q2', 'Q1_div_Q3', 'Q4_div_Q5', 'Q1_absdiff_Q3', 'Q3',
                                  'Q1_plus_Q2', 'Q3_div_Q1', 'Q2_div_Q4', 'Q2_div_Q3', 'Q4_div_Q1',
                                  'Q2_absdiff_Q4'],
        'MI+CHI_reversed': ['Q4_div_Q1', 'Q1_absdiff_Q4', 'Q5', 'Q2_div_Q3', 'Min_Score', 'Q2_div_Q1',
                            'Q4_div_Q5', 'Q3_div_Q1', 'Q2_div_Q5', 'Q4', 'Q4_div_Q2', 'Q2', 'Q2_div_Q4',
                            'Q2_plus_Q5', 'Q1_div_Q4', 'Q1_div_Q2', 'Total_Score'],
        'Fisher_reversed': ['Q4_div_Q2', 'Q2_absdiff_Q5', 'Q1_absdiff_Q2', 'Q1_absdiff_Q4', 'Q1', 'Q1_div_Q4',
                            'Gender', 'Q1_div_Q2', 'Q3', 'Q5', 'Q1_plus_Q5', 'Min_Score', 'Q4', 'Q2',
                            'Q1_plus_Q2', 'Q2_plus_Q5', 'Total_Score',],
        'Embedded RandomForest_reversed': ['Q2_absdiff_Q4', 'Q4_div_Q1', 'Q2_div_Q3', 'Q2_div_Q4',
                                           'Q3_div_Q1', 'Q1_plus_Q2', 'Q3', 'Q1_absdiff_Q3', 'Q4_div_Q5',
                                           'Q1_div_Q3', 'Q4_div_Q2', 'Q2_plus_Q5', 'Q3_absdiff_Q4',
                                           'Q4_div_Q3', 'Std_Dev', 'Gender', 'Total_Score']
    }



    # Initialize results storage
    results = {key: [] for key in feature_sets.keys()}

    # Loop through each feature set
    for feature_set_name, features in feature_sets.items():
        current_features = features.copy()
        while len(current_features) >= 1:
            # Extract the data for the current feature set
            X_raw = data[current_features].values

            # Apply Z-score standardization
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_raw)

            # Calculate the distance ratio
            labels = data[target].values
            distance_ratio = ClusteringEvaluator.distance_ratio(X_processed, labels)

            # Store the result
            results[feature_set_name].append((len(current_features), distance_ratio))

            # Remove the last feature
            current_features.pop()

    # Convert results to a DataFrame for printing
    results_df = pd.DataFrame({
        feature_set_name: [f"{dim}: {ratio:.4f}" for dim, ratio in result]
        for feature_set_name, result in results.items()
    })

    # Print the results table
    print("\n=== Distance Ratio Results ===")
    print(results_df)

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot each feature set's results
    for feature_set_name, result in results.items():
        dimensions = [r[0] for r in result]
        ratios = [r[1] for r in result]
        plt.plot(dimensions, ratios, marker='o', linewidth=2, label=feature_set_name)

    # Chart style settings
    plt.title('Distance Ratio vs Feature Dimension')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Distance Ratio (Lower is Better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.xticks(dimensions)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()



    # Initialize results storage
    results = []

    # Extract the filtered features
    X_raw = data[filtered_features].values

    # Apply Z-score standardization
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_raw)

    # Loop through dimensions from 1 to 15
    for n_components in range(1, 16):
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_standardized)

        # Calculate the distance ratio
        labels = data[target].values
        distance_ratio = ClusteringEvaluator.distance_ratio(X_pca, labels)

        # Store the result
        results.append((n_components, distance_ratio))

    # Convert results to a DataFrame for printing
    results_df = pd.DataFrame(results, columns=['Dimensions', 'Distance Ratio'])

    # Print the results table
    print("\n=== Distance Ratio Results ===")
    print(results_df)

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Extract dimensions and ratios for plotting
    dimensions = results_df['Dimensions']
    ratios = results_df['Distance Ratio']

    plt.plot(dimensions, ratios, marker='o', linewidth=2, label='PCA')

    # Chart style settings
    plt.title('Distance Ratio vs PCA Dimensions')
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Distance Ratio (Lower is Better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.xticks(dimensions)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()