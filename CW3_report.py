import sys
from sklearn.model_selection import train_test_split
import random
from CW_functions import *


warnings.filterwarnings('ignore')
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
random_state = 42
random.seed(random_state)
np.random.seed(random_state)


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

        # Count and display the distribution of students by Programme
        programme_counts = data['Programme'].value_counts()
        print("\n=== Programme Distribution ===")
        for programme, count in programme_counts.items():
            print(f"Programme {programme}: {count} students")

        # Calculate and print percentages
        programme_percentages = programme_counts / len(data) * 100
        print("\n=== Programme Distribution (%) ===")
        for programme, percentage in programme_percentages.items():
            print(f"Programme {programme}: {percentage:.2f}%")
    except FileNotFoundError:
        print("Error: 'training_set.xlsx' not found.")
        sys.exit(1)

    #====================#
    #   FeatureCreation  #
    #====================#

    # original_features = ['Gender', 'Grade', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    original_features = data.iloc[:, 1:].columns.tolist()
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

    # Rank features
    final_ranking = feature_engineer.rank_features(data, filtered_features, target)
    print("Feature Ranking:")
    print(final_ranking)

    # Rank features using Chi-square test
    chi2_ranking = feature_engineer.chi2_feature_ranking(data, filtered_features, target)
    print("\n=== Chi-Square Feature Ranking ===")
    print(chi2_ranking)

    # Rank features using Mutual Information
    mi_ranking = feature_engineer.mutual_info_feature_ranking(data, filtered_features, target)
    print("\n=== Mutual Information Feature Ranking ===")
    print(mi_ranking)

    # You can select top features based on each ranking method
    top_n = 15  # Number of top features to select
    top_chi2_features = chi2_ranking['Feature'].tolist()[:top_n]
    top_mi_features = mi_ranking['Feature'].tolist()[:top_n]
    top_final_ranking = final_ranking['Feature'].tolist()[:top_n]
    print(f"Top {top_n} Chi-Square Features: {top_chi2_features}")
    print(f"Top {top_n} Mutual Information Features: {top_mi_features}")
    print(f"Top {top_n} Final Ranking Features: {top_final_ranking}")

    # Perform similarity-based feature selection
    data_with_target = data[filtered_features].copy()
    data_with_target[target] = data[target]  # Add the target column
    similarity_selected_data, similarity_features = feature_engineer.similarity_based_feature_selection(
        data_with_target, target, top_k=15)
    print(f"Top 15 Features (Similarity-Based): {similarity_features}")

    # Perform Laplacian score-based feature selection
    laplacian_selected_data, laplacian_features = feature_engineer.laplacian_score_feature_selection(
        data[filtered_features], top_k=15)
    print(f"Top 15 Features (Laplacian Score): {laplacian_features}")

    # Use Sequential Forward Selection with multiple models
    data_subset = data[filtered_features].copy()
    X = data_subset
    y = data[target]

    # Perform Sequential Forward Selection
    sfs_results = feature_engineer.perform_sequential_forward_selection(X, y)
    print("\n=== Sequential Forward Selection Results ===")
    for model_name, selected_feats in sfs_results.items():
        print(f"{model_name}: {selected_feats}")
        print(f"Number of features selected: {len(selected_feats)}")

    # Ensure the target column is included in the DataFrame
    data_with_target = data[filtered_features + [target]].copy()

    # Perform embedded feature selection
    selected_data, selected_features = feature_engineer.embedded_feature_selection(data_with_target, target,
                                                                                   max_features=15)
    print(f"Selected Features: {selected_features}")

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
            'Embedded RandomForest': ['Total_Score', 'Gender'],
            'test':[ 'Grade', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'],

    }
    selected_feature_set = 'original(7 features)'  # select feature sets
    selected_features = feature_sets[selected_feature_set]
    X = data[selected_features]
    y = data[target]

    # ====================== #
    # Train-Test Split       #
    # ====================== #
    print("\n=== Splitting Data into Train and Test Sets ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Show class distribution in train set
    train_distribution = pd.Series(y_train).value_counts()
    print("\n=== Class Distribution in Training Set ===")
    for cls, count in train_distribution.items():
        print(f"Programme {cls}: {count} samples ({count / len(y_train) * 100:.2f}%)")

    # ====================== #
    #   Data Preprocessing   #
    # ====================== #

    preprocess_method = 'standard'  # Options: 'none', 'standard', 'minmax', 'robust', 'pca'

    # Preprocess training data
    X_train_processed = feature_engineer.preprocess_features(X_train, method=preprocess_method)
    print(f"Training data preprocessed using '{preprocess_method}' method.")

    # Preprocess test data - important: use the same scaler fit on training data
    X_test_processed = feature_engineer.preprocess_features(X_test, method=preprocess_method)
    print(f"Test data preprocessed using '{preprocess_method}' method.")

    # ============================== #
    #    Hyperparameter Tuning       #
    # ============================== #

    auto_tuner = AutoClassifierTuner(random_state=42, verbose=1)

    # Tuning models with cross-validation on the original training data
    print("\n=== Hyperparameter Tuning with Cross-Validation ===")

    print("\n--- Tuning Gaussian Naive Bayes ---")
    gnb_model, gnb_params, gnb_cv_score = auto_tuner.tune_gnb(X_train_processed, y_train)
    print(f"GNB Best parameters: {gnb_params}")
    print(f"GNB Cross-validation score: {gnb_cv_score:.4f}")

    print("\n--- Tuning KNN ---")
    knn_model, knn_params, knn_cv_score = auto_tuner.tune_knn(X_train_processed, y_train)
    print(f"KNN Best parameters: {knn_params}")
    print(f"KNN Cross-validation score: {knn_cv_score:.4f}")

    print("\n--- Tuning Decision Tree ---")
    dt_model, dt_params, dt_cv_score = auto_tuner.tune_decision_tree(X_train_processed, y_train)
    print(f"DT Best parameters: {dt_params}")
    print(f"DT Cross-validation score: {dt_cv_score:.4f}")

    print("\n--- Tuning Random Forest ---")
    rf_model, rf_params, rf_cv_score = auto_tuner.tune_random_forest(X_train_processed, y_train)
    print(f"RF Best parameters: {rf_params}")
    print(f"RF Cross-validation score: {rf_cv_score:.4f}")

    print("\n--- Tuning Ensemble ---")
    ensemble_model, ensemble_params, ensemble_cv_score = auto_tuner.tune_ensemble(X_train_processed, y_train)
    print(f"Ensemble Best parameters: {ensemble_params}")
    print(f"Ensemble Cross-validation score: {ensemble_cv_score:.4f}")

    # === Tuning Stacking Ensemble Model === #

    # Tune and evaluate stacking model
    print("\n=== Tuning Stacking Ensemble Model ===")
    stacking_model, best_params, best_score = auto_tuner.tune_stacking(X_train_processed, y_train)

    # Print results
    print("\n=== Results for Stacking Ensemble ===")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")

    print("\n--- Tuning XGBoost ---")
    xgb_model, xgb_params, xgb_cv_score = auto_tuner.tune_xgboost(X_train_processed, y_train)
    print(f"XGBoost Best parameters: {xgb_params}")
    print(f"XGBoost Cross-validation score: {xgb_cv_score:.4f}")

    # ============================== #
    # Train Models with Best Params  #
    # ============================== #
    print("\n=== Training Final Models with Best Parameters ===")

    # Manually initialize and train each model using best parameters
    tuner = ManualClassifierTuner(random_state=42)

    # Train GNB with best parameters
    final_gnb = tuner.train_gnb(X_train_processed, y_train, **gnb_params)

    # Train KNN with best parameters
    final_knn = tuner.train_knn(X_train_processed, y_train, **knn_params)

    # Train Decision Tree with best parameters
    final_dt = tuner.train_decision_tree(X_train_processed, y_train, **dt_params)

    # Train Random Forest with best parameters
    final_rf = tuner.train_random_forest(X_train_processed, y_train, **rf_params)

    # Train Ensemble with best parameters
    final_ensemble = ensemble_model

    final_stacking_ensemble = stacking_model
    final_xgb = xgb_model

    # ==============================  #
    # Train Models with Manual Params #
    # ==============================  #
    print("\n=== Training Models with Manually Defined Parameters ===")

    # # Initialize the tuner
    # tuner = ManualClassifierTuner(random_state=42)
    #
    # # Train Gaussian Naive Bayes (GNB) with manually defined parameters
    # print("\n--- Training Gaussian Naive Bayes ---")
    # final_gnb = tuner.train_gnb(
    #     X_train_processed,
    #     y_train,
    #     var_smoothing=0.6579  # Example parameter
    # )
    #
    # # Train K-Nearest Neighbors (KNN) with manually defined parameters
    # print("\n--- Training KNN ---")
    # final_knn = tuner.train_knn(
    #     X_train_processed,
    #     y_train,
    #     n_neighbors=9,
    #     weights='uniform',
    #     algorithm='brute',
    #     p=1,
    #     metric='chebyshev'
    # )
    #
    # # Train Decision Tree (DT) with manually defined parameters
    # print("\n--- Training Decision Tree ---")
    # final_dt = tuner.train_decision_tree(
    #     X_train_processed,
    #     y_train,
    #     criterion='gini',
    #     max_depth=10,
    #     min_samples_split=5
    # )

    # # Train Random Forest (RF) with manually defined parameters
    # print("\n--- Training Random Forest ---")
    # final_rf = tuner.train_random_forest(
    #     X_train_processed,
    #     y_train,
    #     n_estimators=100,
    #     max_depth=15,
    #     min_samples_split=3
    # )
    #
    # # Train Ensemble Model (VotingClassifier) with manually defined weights
    # print("\n--- Training Ensemble Model ---")
    # knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski')
    # gnb = GaussianNB(var_smoothing=1e-8)
    # dt = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=5)
    #
    # final_ensemble = VotingClassifier(
    #     estimators=[('knn', knn), ('gnb', gnb), ('dt', dt)],
    #     voting='soft',
    #     weights=[2, 1, 3]  # Example weights: KNN=2, GNB=1, DT=3
    # )
    #
    # final_ensemble.fit(X_train_processed, y_train)


    # ============================== #
    # Evaluate on Training and Test  #
    # ============================== #
    print("\n=== Model Evaluation ===")

    # Function to evaluate and print metrics


    # Evaluate all models on training set
    print("\n=== TRAINING SET EVALUATION ===")
    gnb_train_metrics = ClassifyingEvaluator.evaluate_and_print(final_gnb, X_train_processed, y_train, "Training")
    knn_train_metrics = ClassifyingEvaluator.evaluate_and_print(final_knn, X_train_processed, y_train, "Training")
    dt_train_metrics = ClassifyingEvaluator.evaluate_and_print(final_dt, X_train_processed, y_train, "Training")
    rf_train_metrics = ClassifyingEvaluator.evaluate_and_print(final_rf, X_train_processed, y_train, "Training")
    ensemble_train_metrics = ClassifyingEvaluator.evaluate_and_print(final_ensemble, X_train_processed, y_train, "Training")
    stacking_train_metrics = ClassifyingEvaluator.evaluate_and_print(final_stacking_ensemble, X_train_processed, y_train, "Training")
    xgb_train_metrics = ClassifyingEvaluator.evaluate_xgb(final_xgb, X_train_processed, y_train, "Training")

    # # Evaluate all models on test set
    print("\n=== TEST SET EVALUATION ===")
    gnb_test_metrics = ClassifyingEvaluator.evaluate_and_print(final_gnb, X_test_processed, y_test, "Test")
    knn_test_metrics = ClassifyingEvaluator.evaluate_and_print(final_knn, X_test_processed, y_test, "Test")
    dt_test_metrics = ClassifyingEvaluator.evaluate_and_print(final_dt, X_test_processed, y_test, "Test")
    rf_test_metrics = ClassifyingEvaluator.evaluate_and_print(final_rf, X_test_processed, y_test, "Test")
    ensemble_test_metrics = ClassifyingEvaluator.evaluate_and_print(final_ensemble, X_test_processed, y_test, "Test")
    stacking_test_metrics = ClassifyingEvaluator.evaluate_and_print(final_stacking_ensemble, X_test_processed, y_test, "Test")
    xgb_test_metrics = ClassifyingEvaluator.evaluate_xgb(final_xgb, X_test_processed, y_test, "Test")

    # ============================== #
    #   Visualize learning curves    #
    # ============================== #

    ClassifyingEvaluator.plot_learning_curve(final_gnb, X_train_processed, y_train, cv=5, scoring='accuracy')
    ClassifyingEvaluator.plot_learning_curve(final_knn, X_train_processed, y_train, cv=5, scoring='accuracy')
    ClassifyingEvaluator.plot_learning_curve(final_dt, X_train_processed, y_train, cv=5, scoring='accuracy')
    ClassifyingEvaluator.plot_learning_curve(final_rf, X_train_processed, y_train, cv=5, scoring='accuracy')
    ClassifyingEvaluator.plot_learning_curve(final_ensemble, X_train_processed, y_train, cv=5, scoring='accuracy')
    ClassifyingEvaluator.plot_learning_curve(final_stacking_ensemble, X_train_processed, y_train, cv=5,
                                                      scoring='accuracy')

    # ============================== #
    #   Visualize confusion matrix   #
    # ============================== #
    # Plot confusion matrices for each model on the test set
    print("\n=== Confusion Matrices for Test Set ===")

    # Gaussian Naive Bayes
    print("\n--- Confusion Matrix for Gaussian Naive Bayes ---")
    gnb_test_predictions = final_gnb.predict(X_test_processed)
    ClassifyingEvaluator.plot_confusion_matrix(
        y_true=y_test,
        y_pred=gnb_test_predictions,
        class_names=np.unique(y_test),  # Replace with actual class names if available
        normalize=True,
        title="Confusion Matrix for Gaussian Naive Bayes"
    )

    # K-Nearest Neighbors
    print("\n--- Confusion Matrix for K-Nearest Neighbors ---")
    knn_test_predictions = final_knn.predict(X_test_processed)
    ClassifyingEvaluator.plot_confusion_matrix(
        y_true=y_test,
        y_pred=knn_test_predictions,
        class_names=np.unique(y_test),  # Replace with actual class names if available
        normalize=True,
        title="Confusion Matrix for K-Nearest Neighbors"
    )

    # Decision Tree
    print("\n--- Confusion Matrix for Decision Tree ---")
    dt_test_predictions = final_dt.predict(X_test_processed)
    ClassifyingEvaluator.plot_confusion_matrix(
        y_true=y_test,
        y_pred=dt_test_predictions,
        class_names=np.unique(y_test),  # Replace with actual class names if available
        normalize=True,
        title="Confusion Matrix for Decision Tree"
    )

    # Random Forest
    print("\n--- Confusion Matrix for Random Forest ---")
    rf_test_predictions = final_rf.predict(X_test_processed)
    ClassifyingEvaluator.plot_confusion_matrix(
        y_true=y_test,
        y_pred=rf_test_predictions,
        class_names=np.unique(y_test),  # Replace with actual class names if available
        normalize=True,
        title="Confusion Matrix for Random Forest"
    )

    # Ensemble Model
    print("\n--- Confusion Matrix for Ensemble Model ---")
    ensemble_test_predictions = final_ensemble.predict(X_test_processed)
    ClassifyingEvaluator.plot_confusion_matrix(
        y_true=y_test,
        y_pred=ensemble_test_predictions,
        class_names=np.unique(y_test),  # Replace with actual class names if available
        normalize=True,
        title="Confusion Matrix for Ensemble Model"
    )

    # Stacking Ensemble Model
    print("\n--- Confusion Matrix for Stacking Ensemble Model ---")
    stacking_test_predictions = final_stacking_ensemble.predict(X_test_processed)
    ClassifyingEvaluator.plot_confusion_matrix(
        y_true=y_test,
        y_pred=stacking_test_predictions,
        class_names=np.unique(y_test),  # Replace with actual class names if available
        normalize=True,
        title="Confusion Matrix for Stacking Ensemble Model"
    )

    #=============================== #
    #       the best model KNN       #
    #=============================== #

    # 使用原始7个特征
    original_features = ['Gender', 'Grade', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    X_original = data[original_features]

    # 重新分割数据
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_original, y, test_size=0.2, random_state=42, stratify=y
    )

    # 预处理数据
    X_train_orig_processed = feature_engineer.preprocess_features(X_train_orig, method=preprocess_method)
    X_test_orig_processed = feature_engineer.preprocess_features(X_test_orig, method=preprocess_method)

    # 使用指定参数创建并训练KNN模型
    knn_params = {
        'algorithm': 'brute',
        'metric': 'chebyshev',
        'n_neighbors': 9,
        'p': 1,
        'weights': 'uniform'
    }
    knn_model = KNeighborsClassifier(**knn_params)
    knn_model.fit(X_train_orig_processed, y_train_orig)

    # 在测试集上评估模型并打印结果
    print("\n=== KNN with original 7 features on TEST SET ===")
    knn_test_metrics = ClassifyingEvaluator.evaluate_and_print(
        knn_model, X_test_orig_processed, y_test_orig, "Test"
    )

    # 绘制混淆矩阵
    print("\n--- Confusion Matrix for KNN (original 7 features) ---")
    knn_test_predictions = knn_model.predict(X_test_orig_processed)
    ClassifyingEvaluator.plot_confusion_matrix(
        y_true=y_test_orig,
        y_pred=knn_test_predictions,
        class_names=np.unique(y_test_orig),
        normalize=True,
        title="Confusion Matrix for KNN (original 7 features)"
    )



if __name__ == "__main__":
    main()


