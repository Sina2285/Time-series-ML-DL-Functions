#make sure to restart kernel in lab notebook before importing. 

def df_abundance_nugent(filename=None, log=False, rel_abund=False, zscore=False, cst=False, metabolomics=False): 
    """
    Sina Bonakdar March, 2025
    Load data from a specified Excel file into a pandas DataFrame.
    Uses a default file path if no filename is provided.

    Parameters:
    filename (str): The path to the Excel file.
    log=True: perform natural log transformation
    rel_abund=True: perform relative abundance data
    zscore=True: preprocess of relative abundance using zscore
    cst=True: Will record CST-I,II,III,IV,V as a column as well
    metabolomics=True: include the column of metabolic data (important to know how many days for each block we will have metabolic data

    Returns:
    Cleaned dataframe with 4751 rows and 83 columns 
    including bacterial data + NUGENT Class for each timepoint and sample. 
    """
    import pandas as pd 
    from sklearn import preprocessing
    import numpy as np
    if not filename:
        filename = '/Users/bonakdar/Desktop/Arnold Lab/QE/HMP/HMP_ALL_DATA_Release2_with_METABOLOMICS_ALL_12172017.xlsx'
    # Load your DataFrame
    df = pd.read_excel(filename)
    #drop the samples with mensturation for all time points 
    values_to_drop = [4, 19, 63, 79]
    df = df[~df['PID'].isin(values_to_drop)]
    # Determine the indices of columns to drop
    cols_to_drop = list((0,1,3,14))
    if metabolomics==True:
        cols_to_drop += list(range(5, 8))
        cols_to_drop += list(range(9, 13))
    else:
        cols_to_drop += list(range(5, 13))
    if cst==True:
        cols_to_drop += list(range(16, 145))
    else:
        cols_to_drop += list(range(16, 146))
    cols_to_drop += list(range(-8, 0))
    df.drop(df.columns[cols_to_drop], axis=1, inplace=True)
    #Take the label column named (NUGENT_CLASS) to the end: 
    column_to_move = 'NUGENT_CLASS'
    columns = list(df.columns)
    columns.append(columns.pop(columns.index(column_to_move)))
    df = df[columns]
    #drop the columns with more than 3800 nan values: different number of rows will be dropped by including vs. not including CSTs
    nan_counts = df.isna().sum() 
    cols_to_drop = nan_counts[nan_counts >= 3800].index
    if cst==True:
        df.drop(columns=cols_to_drop[2:], inplace=True) #drop all columns except menstruation 
    else:
        df.drop(columns=cols_to_drop[1:], inplace=True)
    #drop rows with nan values
    ignore_columns = ['PID', 'MENSTRUATION', 'SERIAL']
    df = df.dropna(subset=[col for col in df.columns if col not in ignore_columns])
    df.reset_index(drop=True, inplace=True)
    if cst==True and metabolomics==True:
        bacteria_columns = df.columns[5:-1]
    elif cst==True and metabolomics==False:
        bacteria_columns = df.columns[4:-1]
    elif cst==False and metabolomics==True:
        bacteria_columns = df.columns[4:-1]
    elif cst==False and metabolomics==False:
        bacteria_columns = df.columns[3:-1]
    if log==True:
        #perform log transformation
        small_number = 1e-10
        df[bacteria_columns] = df[bacteria_columns].replace(0, small_number)
        df[bacteria_columns] = np.log(df[bacteria_columns])
        print('log transformed data')
        return df   
    if rel_abund==True:
        #create relative abundance data 
        row_sums = df[bacteria_columns].sum(axis=1) # Calculate the sum of the relevant columns for each row
        df[bacteria_columns] = df[bacteria_columns].div(row_sums, axis=0)
        if zscore==True:
        #preprocessing of df: standardize the data for each row. Makes each sample to have zero mean and 1 std. Known as zscore  normalization 
            for col in bacteria_columns:
                df[col] = preprocessing.scale(df[col].values)  # Scale the data to make it between 0-1.
            print('relative abundance + zscore data')
            return df
        else:
            print('relative abundance data')
            return df      
    else:
        print('either log, rel_abund, or zscore has to be True. This is original data')
        return df
        



def menstruation_xblock_y_premensesX(df,premenses_included=False,menses_included=False,four_labels=False, multistability=False):
    """
    Sina Bonakdar July, 2025
    Using the cleaned dataframe from df_abundance_Nug or Ams  as input we create xblock and y labels
    xblock:
    if menses_included=True and premenses_included=True: xblock: premenses+menses
    ylabel: depends on we want 2 labels or 4 labels;
    2 labels: using post menses for each menstruation: if BV occurs on any day: y=1
    else: y=0
    4 labels: considering both premenses and post menses data: 
    premenses: no_BV, post_menses: no_BV: healthy->healthy (label=0)
    premenses: no_BV, post_menses: has BV: healthy->BV (label=1)
    premenses: has BV, post_menses: no_BV: BV->healthy (label=2)
    premenses: has BV, post_menses: has BV: BV->BV (label=3)

    multistability==True: 
    label = 0 if BV doesn't occur on any days post menses
    label = 1 if BV occurs on all days post menses
    label = 2 if BV occurs on some days post menses (We have both BV and non BV days)
    """
    import pandas as pd
    import numpy as np
    DF = df.copy() #create a copy to don't update original df
    # Columns you want to drop later
    if 'NUGENT_CLASS' in DF.columns: #In nugent, label column is NUGENT_CLASS, make it similar to Amsel data
        DF = DF.rename(columns={'NUGENT_CLASS': 'label'})
    # Columns you want to drop later
    columns_to_drop = ['PID', 'MENSTRUATION', 'label', 'SERIAL']
    X = []
    y = []
    # Replace all numeric values in the 'MENSTRUATION' column with 1, keeping NaN unchanged
    DF['MENSTRUATION'] = DF['MENSTRUATION'].apply(lambda x: 1 if pd.notna(x) else np.nan)
    # Iterate over each participant ID's data
    for pid, group in DF.groupby('PID'):
        group = group.sort_values(by='SERIAL')  # Ensure correct temporal ordering
        # Identify the start of each menstruation period
        menstruation_starts = group.index[
            group['MENSTRUATION'].notna() & group['MENSTRUATION'].shift(1).isna()
        ]
        current_start = 0  # For pre-menstruation tracking
        for i, start_idx in enumerate(menstruation_starts):
            # This ensures we handle the last section correctly
            next_start_idx = menstruation_starts[i + 1] if i + 1 < len(menstruation_starts) else None
            # Pre-menstruation: Capture timepoints before this cycle if any
            pre_menstruation = group.loc[current_start:start_idx - 1, :]
            # Determine menstruation phase
            menstruation = group.loc[start_idx:]
            menstruation_end = menstruation[menstruation['MENSTRUATION'].isna()].index.min()
            menstruation = menstruation.loc[:menstruation_end - 1, :] if pd.notna(menstruation_end) else menstruation
            # Determine post-menstruation segment for BV classification
            if pd.notna(menstruation_end):
                # End at the start of next menstruation or end of the group
                post_menstruation_end = next_start_idx if next_start_idx is not None else len(group)+menstruation_end
                post_menstruation = group.loc[menstruation_end:post_menstruation_end - 1, 'label']
                #print('post menstruation labels: ',post_menstruation)
                if four_labels==True:
                    has_bv_post = 'BV' in post_menstruation.values
                    has_bv_pre = 'BV' in pre_menstruation.values
                    if not has_bv_pre and not has_bv_post:
                        class_label=0
                    if not has_bv_pre and has_bv_post:
                        class_label=1
                    if has_bv_pre and not has_bv_post:
                        class_label=2
                    if has_bv_pre and has_bv_post:
                        class_label=3

                #Check multistability: 
                if multistability==True:
                    if 'BV' not in post_menstruation.values:
                        class_label = 0
                    elif post_menstruation.eq('BV').all():
                        class_label = 1
                    else:
                        class_label = 2     
                    
                # Assign class label only if post-menstruation data is present
                if four_labels==False and multistability==False:
                    has_bv = 'BV' in post_menstruation.values
                    class_label = 1 if has_bv else 0
                # Form the x_block if both menstruation and post-menstruation data exist
                if not menstruation.empty:
                    #x_block = pd.concat([pre_menstruation, menstruation])
                    if premenses_included==True and menses_included==False:
                        x_block = pd.concat([pre_menstruation]) #We use only premenses data in xblock
                    elif premenses_included==True and menses_included==True:
                        x_block = pd.concat([pre_menstruation, menstruation])
                    elif premenses_included==False and menses_included==True: 
                        x_block = pd.concat([menstruation])
                    elif premenses_included==False and menses_included==False:
                        #return None, None
                        raise ValueError("Either premenses, menses, or both have to be True.")
                    x_block = x_block.drop(columns=columns_to_drop)
                    #print(x_block)
                    X.append(x_block)
                    y.append(class_label)
                # Update the start for the next pre-menstruation phase
                current_start = menstruation_end
            else:
                current_start = start_idx  # Move to the start of the next possible cycle

    # Convert lists to numpy arrays; Use dtype=object for varying lengths
    X = np.array(X, dtype=object)
    y = np.array(y)
    print(f"Number of x_blocks (X): {len(X)}")
    print(f"Shape of labels (y): {y.shape}")
    return X, y


def drop_blocks(X,y,thres=None):
    """
    Sina Bonakdar March, 2025
    Drop xblocks and correpsonding yvalues if they had
    less than a threshold timepoints
    """
    import numpy as np 
    import pandas as pd
    if thres==None:
        thres = 4
    # Initialize lists for filtered sequences and labels
    X_filtered = []
    y_filtered = []
    # Iterate over the sequences and their corresponding labels
    for x_block, label in zip(X, y):
        if len(x_block) >= thres:  # Check if the sequence has more timepoints
            X_filtered.append(x_block)
            y_filtered.append(label)
    # Convert to numpy arrays if needed (dtype=object ensures support for varying sequence lengths)
    X_filtered = np.array(X_filtered, dtype=object)
    y_filtered = np.array(y_filtered)
    # Check the results
    print(f"Filtered number of x_blocks: {len(X_filtered)}")
    print(f"Filtered shape of labels (y): {y_filtered.shape}")
    # Verify distribution
    print(f"Number of 1s: {np.sum(y_filtered == 1)}")
    print(f"Number of 0s: {np.sum(y_filtered == 0)}")
    if len(np.unique(y_filtered))>2:
        print(f"Number of 2s: {np.sum(y_filtered == 2)}")
        print(f"Number of 3s: {np.sum(y_filtered == 3)}")
    return X_filtered, y_filtered


def lstm_model(X, y, padding_value=None, n_folds=None):
    """
    Sina Bonakdar Sep, 2025
    LSTM model with 3 layers and cross-validation.
    Returns final trained model and average metrics.
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization, Masking
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, roc_auc_score

    if n_folds is None:
        n_folds = 5

    # --- Ensure X is numpy with 3D shape ---
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
        X = np.expand_dims(X, axis=1)  
    elif isinstance(X, np.ndarray):
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)

    max_length = max(len(seq) for seq in X)

    if padding_value is not None:
        X = pad_sequences(X, maxlen=max_length, dtype='float32', padding='post', value=padding_value)

    y = np.array(y)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "specificity": [], "mse": [], "roc_auc": []}
    final_model = None

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        print(f"\nTraining fold {fold}...")

        train_x, validation_x = X[train_index], X[test_index]
        train_y, validation_y = y[train_index], y[test_index]

        model = Sequential()
        if padding_value is not None:
            model.add(Masking(mask_value=padding_value, input_shape=(max_length, X.shape[2])))     
        else:
            model.add(tf.keras.Input(shape=(max_length, X.shape[2])))

        # LSTM layers
        model.add(LSTM(units=256, activation='tanh', return_sequences=True))
        model.add(BatchNormalization()); model.add(Dropout(0.5))
        model.add(LSTM(units=256, activation='tanh', return_sequences=True))
        model.add(BatchNormalization()); model.add(Dropout(0.5))
        model.add(LSTM(units=256, activation='tanh'))
        model.add(BatchNormalization()); model.add(Dropout(0.5))

        # Dense layers
        model.add(Dense(32, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        opt = tf.keras.optimizers.RMSprop(learning_rate=1e-3, decay=1e-5)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

        model.fit(train_x, train_y, epochs=500, validation_data=(validation_x, validation_y), 
                  callbacks=[early_stopping], verbose=0)

        y_pred_prob = model.predict(validation_x)
        y_pred = np.argmax(y_pred_prob, axis=1)

        acc = accuracy_score(validation_y, y_pred)
        prec = precision_score(validation_y, y_pred, zero_division=0)
        rec = recall_score(validation_y, y_pred, zero_division=0)
        f1 = f1_score(validation_y, y_pred, zero_division=0)
        mse = mean_squared_error(validation_y, y_pred)

        cm = confusion_matrix(validation_y, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            spec = 0

        try:
            roc_auc = roc_auc_score(validation_y, y_pred_prob[:, 1])
        except ValueError:
            roc_auc = np.nan

        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        metrics["specificity"].append(spec)
        metrics["mse"].append(mse)
        metrics["roc_auc"].append(roc_auc)

        print(f"Fold {fold} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, "
              f"F1: {f1:.4f}, Spec: {spec:.4f}, MSE: {mse:.4f}, AUC: {roc_auc:.4f}")

        final_model = model  # keep last trained model

    avg_metrics = {m: np.nanmean(v) for m, v in metrics.items()}

    print("\n=== Average Metrics Across Folds ===")
    for k, v in avg_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    return final_model, avg_metrics



def flatten(X):
    """
    Sina Bonakdar March, 2025
    Flatten the X for using in ML. 
    Here for each block we use the last timepoints 
    minimum number of timepoints between all features 
    is the number of timepoints we use for each block
    then flat the data
    """
    import pandas as pd
    import numpy as np
    # Calculate the number of features (assuming the same for each block)
    num_features = X[0].shape[1]
    # List to store the arrays of last valid timepoints for each block
    last_valid_timepoints = []
    # Determine the minimum number of time points across all blocks
    min_timepoints = min(block.shape[0] for block in X)
    for block in X:
        # Take the last `min_timepoints` rows from each block
        valid_area = block.to_numpy()[-min_timepoints:, :]
        # Flatten this valid area
        flattened_valid_area = valid_area.flatten()
        # Append the flattened array to the list
        last_valid_timepoints.append(flattened_valid_area)
    # Generate column names for the final DataFrame
    columns = []
    for t in range(min_timepoints):
        columns.extend([f"{feature}_t{t+1}" for feature in X[0].columns])
    # Create the final DataFrame with selected last valid time points
    DF = pd.DataFrame(last_valid_timepoints, columns=columns)
    return DF

def random_forest_model(df,y):
    """
    Sina Bonakdar March, 2025
    A simple RF model with stratified 5-fold CV
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import numpy as np
    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators: number of trees, 
    # Create StratifiedKFold object for 5-fold cross-validation
    stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Perform cross-validation and compute scores
    cv_scores = cross_val_score(rf_classifier, df, y, cv=stratified_kf, scoring='accuracy')
    # Print the cross-validation scores
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation accuracy: {np.mean(cv_scores):.2f}")

def RForest_top_features_plot(df,y,numb_features=None):
    """
    Sina Bonakdar March, 2025
    plot and store the most important features using RF
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import matplotlib.pyplot as plt
    if numb_features==None:
        numb_features=20
    # Initialize and fit the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(df, y)
    # Retrieve feature importances
    importances = rf_classifier.feature_importances_
    # Create a DataFrame for easy viewing and sorting
    feature_importances = pd.DataFrame({
        'Feature': df.columns,
        'Importance': importances
    })
    # Sort the DataFrame by importance
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    # Sort the DataFrame by importance and select top X features
    top_features = feature_importances.sort_values(by='Importance', ascending=False).head(numb_features)
    # Plot the top 20 feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Top {} Feature Importances".format(numb_features))
    plt.barh(top_features['Feature'], top_features['Importance'], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Relative Importance")
    plt.show()
    # Display the sorted feature importances
    return feature_importances


def plsda_model(df,y):
    """
    Sina Bonakdar March, 2025
    A simple plsda model with stratified 5-fold CV
    """
    import pandas as pd
    import numpy as np
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    # Encode the target values using LabelBinarizer for PLS
    lb = LabelBinarizer()
    y_binarized = lb.fit_transform(y)
    # Initialize PLS-DA model
    plsda = PLSRegression(n_components=2)  # You can fine-tune the number of components
    # Stratified 5-fold cross-validation setup
    strat_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in strat_kf.split(df, y):
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = y_binarized[train_index], y[test_index]  
        # Fit the PLS-DA model
        plsda.fit(X_train, y_train)
        # Predict on the test data
        y_pred_binarized = plsda.predict(X_test)
        # Transform back to original labels
        y_pred = lb.inverse_transform(y_pred_binarized)
        # Calculate accuracy for the current fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    # Calculate average accuracy across folds
    mean_accuracy = np.mean(accuracies)
    print(f"Cross-validated accuracies: {accuracies}")
    print(f"Mean cross-validated accuracy: {mean_accuracy:.2f}")
    return plsda


def knn_model(df,y,n_neighbors=None):
    """
    Sina Bonakdar March, 2025
    Simple KNN with 5-fold CV
    """
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    if n_neighbors==None:
        n_neighbors=5
    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)  
    # Stratified 5-fold cross-validation setup
    strat_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in strat_kf.split(df, y):
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fit the KNN model
        knn.fit(X_train, y_train)
        # Predict on the test data
        y_pred = knn.predict(X_test)
        # Calculate accuracy for the current fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    # Calculate average accuracy across folds
    mean_accuracy = np.mean(accuracies)
    print(f"Cross-validated accuracies: {accuracies}")
    print(f"Mean cross-validated accuracy: {mean_accuracy:.2f}")


def yblock_generator(df):
    """
    Sina Bonakdar March, 2025
    Function for one-hot encoding of y: create yblock for use in matlab
    """
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    import numpy as np
    y= pd.DataFrame(df, columns=['Label'])
    # Encode class labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Number of classes
    num_classes = len(np.unique(y_encoded))
    # Ensure the target is in the correct shape (One Hot Encoding for the PLS Regression)
    y_dummy = np.eye(num_classes)[y_encoded]
    # Convert the array to a DataFrame with two columns
    yblock = pd.DataFrame(y_dummy, columns=['Class1', 'Class2'])
    return yblock


def matlab_ddm_save_data(df,y):
    """
    Sina Bonakdar March, 2025
    Save xblock, y, patients,xnames
    """
    import numpy as np
    import pandas as pd
    patients = pd.DataFrame(df.index.astype(str), columns=['Index'])
    patients.to_csv('/Users/bonakdar/Desktop/patients.csv')
    #create xblock 
    df.to_csv('/Users/bonakdar/Desktop/xblock.csv')
    #create xnames
    df.columns.to_frame().T.to_csv('/Users/bonakdar/Desktop/xnames.csv')
    #create yblock
    yblock = yblock_generator(y)
    yblock.to_csv('/Users/bonakdar/Desktop/yblock.csv')



def pca_scores_loading(df,y,numb_loadings=None):
    """
    Sina Bonakdar March, 2025
    PCA scores and loading plot
    numb_loadings = number of toppositive or negative features to plot
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    if numb_loadings==None:
        numb_loadings=10
    pca = PCA(n_components=2)
    pca = pca.fit(df)
    y_score= pca.fit_transform(df)
    print('explained variance: ', pca.explained_variance_ratio_)
    feature_names = df.columns
    loadings = pca.components_
    # Function to get top 10 positive and negative loadings
    def get_top_loadings(loading_values, num=numb_loadings):
        sorted_indices_pos = loading_values.argsort()[-num:][::-1]  # Top positive
        sorted_indices_neg = loading_values.argsort()[:num]         # Top negative
        return sorted_indices_neg.tolist(), sorted_indices_pos.tolist()
    # Select features for PC1
    neg_indices_pc1, pos_indices_pc1 = get_top_loadings(loadings[0])
    key_features_pc1 = neg_indices_pc1 + pos_indices_pc1
    # Select features for PC2
    neg_indices_pc2, pos_indices_pc2 = get_top_loadings(loadings[1])
    key_features_pc2 = neg_indices_pc2 + pos_indices_pc2
    # Set up a 2x2 plotting area
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Plot (1, 1) - Scores plot
    sns.scatterplot(x=y_score[:, 0], y=y_score[:, 1], hue=y, ax=axes[0, 0])
    axes[0, 0].set_title('PCA Scores Plot')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    # Plot (1, 2) - Loadings plot for PC2
    sns.barplot(x=feature_names[key_features_pc2], y=loadings[1][key_features_pc2], ax=axes[0, 1])
    axes[0, 1].set_title('Top {} Positive and Negative Loadings for PC2'.format(numb_loadings))
    axes[0, 1].set_xlabel('Features')
    axes[0, 1].set_ylabel('Loading Value')
    axes[0, 1].set_xticklabels(feature_names[key_features_pc2], rotation=45, ha='right')
    # Plot (2, 1) - Loadings plot for PC1
    sns.barplot(x=loadings[0][key_features_pc1], y=feature_names[key_features_pc1], ax=axes[1, 0])
    axes[1, 0].set_title('Top {} Positive and Negative Loadings for PC1'.format(numb_loadings))
    axes[1, 0].set_xlabel('Loading Value')
    axes[1, 0].set_ylabel('Features')
    # Hide the empty subplot (2, 2)
    axes[1, 1].axis('off')
    # Adjust layout to handle overlapping plots
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)
    plt.show()


def read_df_y(path_df=None,path_y=None):
    """ 
    Sina Bonakdar March, 2025
    read df=X and y and prepare for use in ML_model_evaluations
    default is from NUGENT file: HMP data for MATLAB_Nugent in HMP
    """
    import pandas as pd
    import numpy as np
    if path_df==None:
        path_df='/Users/bonakdar/Desktop/Arnold Lab/QE/HMP/HMP data for MATLAB_Nugent/xblock.csv'
    if path_y==None:
        path_y='/Users/bonakdar/Desktop/Arnold Lab/QE/HMP/HMP data for MATLAB_Nugent/yblock.csv'
    df = pd.read_csv(path_df)
    df = df.iloc[:,1:]
    y = pd.read_csv(path_y)
    y = y.iloc[:,2:]
    y = np.array(y).reshape(-1,)
    return df, y


def plsda_scores_loading(df,y,numb_loadings=None, orthogonalize=False):
    """
    Sina Bonakdar April, 2025
    PLSDA scores and loading plot
    numb_loadings = number of top positive or negative features to plot
    return the scores and loadings values + plot
    Updated to perform orthogonolized model using OPLS package if orthogonalize==True
    For more information visit https://github.com/BiRG/pyopls
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelBinarizer
    from pyopls import OPLS
    if numb_loadings==None:
        numb_loadings=10
    # Standardize the features
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(df)
    X_scaled = df
    # Binarize the output labels for PLS-DA
    lb = LabelBinarizer()
    Y_binarized = lb.fit_transform(y)
    #Orthogonolized
    if orthogonalize==True:
        opls = OPLS(n_components=1)
        X_scaled = opls.fit_transform(X_scaled,Y_binarized)
    # Fit the PLS-DA model
    pls_da = PLSRegression(n_components=2)
    pls_da.fit(X_scaled, Y_binarized)
    # Compute the scores
    scores = pls_da.transform(X_scaled)
    # Extract loadings
    loadings = pls_da.x_loadings_
    # Create a DataFrame of loadings for easier handling
    feature_names = df.columns
    loading_df = pd.DataFrame(loadings.T, columns=feature_names, index=['PLS1', 'PLS2'])
    # Function to get top X positive and negative loadings
    def get_top_loadings(loading_values, num=numb_loadings):
        sorted_indices_pos = loading_values.argsort()[-num:][::-1]  # Top positive
        sorted_indices_neg = loading_values.argsort()[:num]         # Top negative
        return sorted_indices_neg.tolist() + sorted_indices_pos.tolist()
    # Select features for PLS1 and PLS2
    key_features_pls1 = get_top_loadings(loadings.T[0])
    key_features_pls2 = get_top_loadings(loadings.T[1])
    # Set up a 2x2 plotting area
    fig, axes = plt.subplots(2, 2, figsize=(14, 30))
    # Plot (1, 1) - Scores plot
    sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=y, ax=axes[0, 0])
    axes[0, 0].set_title('PLS-DA Scores Plot')
    axes[0, 0].set_xlabel('PLS1')
    axes[0, 0].set_ylabel('PLS2')
    axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    # Set ticks to point inside the plot
    axes[0, 0].tick_params(axis='both', direction='in')
    # Plot (1, 2) - Loadings plot for PLS2
    sns.barplot(x=feature_names[key_features_pls2], y=loadings.T[1, key_features_pls2], ax=axes[0, 1])
    axes[0, 1].set_title('Top {} Positive and Negative Loadings for PLS2'.format(numb_loadings))
    axes[0, 1].set_xlabel('Features')
    axes[0, 1].set_ylabel('Loading Value')
    axes[0, 1].set_xticklabels(feature_names[key_features_pls2], rotation=45, ha='right')
    # Plot (2, 1) - Loadings plot for PLS1
    sns.barplot(x=loadings.T[0, key_features_pls1], y=feature_names[key_features_pls1], ax=axes[1, 0])
    axes[1, 0].set_title('Top {} Positive and Negative Loadings for PLS1'.format(numb_loadings))
    axes[1, 0].set_xlabel('Loading Value')
    axes[1, 0].set_ylabel('Features')
    # Hide the empty subplot (2, 2)
    axes[1, 1].axis('off')
    # Adjust layout to handle overlapping plots
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)
    plt.show()
    return scores, loadings



from sklearn.base import BaseEstimator, ClassifierMixin
class PLS_DA(BaseEstimator, ClassifierMixin): # Define a PLS-DA classifier
    def __init__(self, n_components=2):
        from sklearn.cross_decomposition import PLSRegression
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)
        
    def fit(self, X, y):
        from sklearn.utils.multiclass import unique_labels
        from pyopls import OPLS
        self.classes_ = unique_labels(y)
        self.pls.fit(X, y)
        return self
        
    def predict(self, X):
        predictions = self.pls.predict(X)
        return (predictions >= 0.5).astype(int) if len(self.classes_) == 2 else self._select_class(predictions)
    
    def decision_function(self, X):
        return self.pls.predict(X)
    
    def _select_class(self, predictions):
        return np.argmax(predictions, axis=1)

def specificity_score(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    import numpy as np
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        # Handle a case where the confusion matrix is not 2x2, usually with imbalanced data
        return np.nan  # Use NaN or some other indication of undefined specificity
    tn, fp, fn, tp = cm.ravel()
    if (tn + fp) == 0:
        return np.nan  # Avoid division by zero
    return tn / (tn + fp)




def cross_val_metrics(model, X, y, n_splits, test=True):
    """
    Sina Bonakdar March, 2025
    Calculate cross-validated metrics for a given model.
    
    Parameters:
    - model: The machine learning model to evaluate.
    - X: DataFrame of features.
    - y: Series or array of target values.
    - n_splits: int, number of folds in stratified k-fold.
    - test: boolean, if True, evaluates on test data; if False, on train data.
    
    Returns:
    - averaged_metrics: dict, averaged metrics across folds.
    """
    from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
    mean_squared_error,
    auc,
    precision_recall_curve)
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 
               'specificity': [], 'mse': [], 'auc': []}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Choose which data to evaluate: train or test
        if test:
            X_eval, y_eval = X_test, y_test
        else:
            X_eval, y_eval = X_train, y_train
        
        # Predict on the selected set
        y_pred = model.predict(X_eval)
        
        # Calculate scores for AUC if possible
        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_eval)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_eval)
        
        # Calculate the metrics
        metrics['accuracy'].append(accuracy_score(y_eval, y_pred))
        metrics['precision'].append(precision_score(y_eval, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_eval, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_eval, y_pred, zero_division=0))
        metrics['specificity'].append(specificity_score(y_eval, y_pred))
        metrics['mse'].append(mean_squared_error(y_eval, y_pred))
        if y_score is not None:
            precision, recall, _ = precision_recall_curve(y_eval, y_score)
            metrics['auc'].append(auc(recall, precision))
    
    # Compute averages of the metrics
    averaged_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    return averaged_metrics


def all_model_evaluation(df,y,test=True,n_splits=5):
    """
    Sina Bonakdar March, 2025
    Evaluate all models using StratifiedKFold (5-folds)
    if test==True: model results are based on test data
    if test==False: model results are based on train data
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.dummy import DummyClassifier
    import numpy as np
    import pandas as pd
    y = pd.DataFrame(y)
    X = df
    results = {}
    # Define models to evaluate
    models = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=10000),
        'Linear SVM': SVC(kernel='linear', probability=True),
        'RBF SVM': SVC(kernel='rbf', C=1, probability=True, class_weight='balanced'),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'PLS-DA': PLS_DA() #from PLS_DA class
    }
    for model_name, model in models.items():
        if test==True:
            results[model_name] = cross_val_metrics(model, X, y, n_splits=n_splits,test=True)
        elif test==False:
            results[model_name] = cross_val_metrics(model, X, y, n_splits=n_splits,test=False)
    # Add a baseline with random guessing (Random guessing has precision and recall close to the prevalence of the positive class)
    dummy_model = DummyClassifier(strategy="most_frequent")
    baseline_metrics = cross_val_metrics(dummy_model, X, y, n_splits=n_splits)
    results['Random Guess'] = baseline_metrics
    # Convert the results to a DataFrame for better readability
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(2)
    return results_df

def plot_plsda_scores(scores, y, title='PLS-DA Scores Plot'):
    """
    Sina Bonakdar March, 2025
    plsda scores plot only
    inputs: scores from plsda_scores_loading function,
    y from read_df_y function
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # Flatten y to ensure it's a 1D array
    y = np.array(y)
    y = y.ravel()
    plt.figure(figsize=(8, 6))
    for class_value in np.unique(y):
        # Mask to select scores corresponding to the current class
        mask = y == class_value
        plt.scatter(scores[mask, 0], scores[mask, 1], label=f'Class {class_value}', alpha=0.7)
    plt.title(title)
    plt.xlabel('PLS Component 1')
    plt.ylabel('PLS Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()



from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.multiclass import unique_labels
import numpy as np
class PLS_DA_3class(BaseEstimator, ClassifierMixin):
    """
    Sina Bonakdar March, 2025
    # PLSDA for 3 classes 
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)
    
    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.encoder = OneHotEncoder()
        Y_encoded = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()
        self.pls.fit(X, Y_encoded)
        return self
        
    def predict(self, X):
        predictions = self.pls.predict(X)
        # Convert predictions into class labels by choosing the class with highest probability
        class_indices = np.argmax(predictions, axis=1)
        return self.encoder.inverse_transform(class_indices.reshape(-1, 1)).ravel()
    
    def decision_function(self, X):
        return self.pls.predict(X)


def function_vip(X,y,model,thres=None): 
    """
    Sina Bonakdar March, 2025
    Reference: code from MATLAB: Tahir Mehmood et al, Chemometrics and Intelligent Laboratory Systems 118 (2012)62?69
    Function for creating the VIP scores from PLSDA model
    X: predictors 
    y: response variable
    T: x_scores 
    W: x_weights
    Q: y-loadings 
    VIP=sqrt(p*q/s)
    thres: threshold for returning the top features (e.g. return features with VIP>1)
    Output: vip_features: features with VIP values larger than three;
    vip_df: all features with their VIP values
    """
    if thres==None:
        thres=1
    import numpy as np 
    import pandas as pd
    T = model.x_scores_
    W = model.x_weights_
    Q = model.y_loadings_
    s = np.diag(T.T @ T @ Q.T @ Q) 
    #initializing 
    m,p = X.shape
    m,h = T.shape
    #calculate VIP
    VIP = [] 
    for i in range(0,p):
        weight = []
        for j in range(0,h):
            weight.append((W[i,j] / np.linalg.norm(W[:,j]))**2)

        q=s.T @ weight
        VIP.append(np.sqrt(p*q/np.sum(s)))

    feature_names = X.columns
    vip_df = pd.DataFrame(VIP, index=feature_names, columns=['VIP'])
    vip_features = vip_df[vip_df['VIP']>thres].index
    return vip_features, vip_df



def train_and_plot_decision_tree(df, y, min_samples_split=5, min_samples_leaf=5, ccp_alpha=0.01, max_depth=4):
    """
    Sina Bonakdar, March 2025
    Train and visualize a decision tree, and print the training and validation accuracies.
    
    Parameters:
    - df: DataFrame containing the feature data
    - y: Series or DataFrame containing the target data
    - min_samples_split: The minimum number of samples required to split an internal node
    - min_samples_leaf: The minimum number of samples required to be at a leaf node
    - ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning
    - max_depth: The maximum depth of the tree

    Prints average accuracies and plots the decision tree.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    Y = pd.DataFrame(y)
    # Initialize the decision tree classifier with provided hyperparameters
    tree = DecisionTreeClassifier(random_state=42,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  ccp_alpha=ccp_alpha,
                                  max_depth=max_depth)
    # Initialize stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Lists to keep track of the accuracies
    train_accuracies = []
    val_accuracies = []
    # Perform stratified k-fold cross-validation
    for train_index, val_index in skf.split(df, y):
        X_train, X_val = df.iloc[train_index], df.iloc[val_index]
        y_train, y_val = Y.iloc[train_index], Y.iloc[val_index]
        # Fit the model on the training data
        tree.fit(X_train, y_train)
        # Predict and calculate accuracy for the training data
        y_train_pred = tree.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)
        # Predict and calculate accuracy for the validation data
        y_val_pred = tree.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_accuracies.append(val_accuracy)
    # Report the average accuracy across the 5 folds
    print('Decision Tree Model: ')
    print('Training accuracy per fold:', train_accuracies)
    print('Validation accuracy per fold:', val_accuracies)
    print(f'Average training accuracy: {np.mean(train_accuracies):.2f}')
    print(f'Average validation accuracy: {np.mean(val_accuracies):.2f}')
    # Train the final model on the entire dataset
    tree.fit(df, Y)
    # Visualize the decision tree
    plt.figure(figsize=(60, 30))
    plot_tree(tree, filled=True, feature_names=df.columns, rounded=True, fontsize=28)
    plt.show()


def train_and_plot_decision_tree_with_grid_search(df, y):
    """
    Sina Bonakdar March 2025
    Train and visualize a decision tree using grid search to optimize parameters,
    and print the training and validation accuracies.
    
    Parameters:
    - df: DataFrame containing the feature data
    - y: Series or DataFrame containing the target data
    
    Prints average accuracies and plots the decision tree.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.metrics import accuracy_score, make_scorer
    # Define the parameter grid to search
    param_grid = {
        'min_samples_split': [2, 5, 7, 10],
        'min_samples_leaf': [2, 5, 7, 10],
        'ccp_alpha': [0.0, 0.01, 0.1],
        'max_depth': [None, 3, 4, 5]
    }
    # Initialize the decision tree classifier
    tree = DecisionTreeClassifier(random_state=42)
    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, 
                               scoring=make_scorer(accuracy_score), 
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                               n_jobs=-1)
    # Perform the grid search
    grid_search.fit(df, y)
    # Get the best parameters and maximum cross-validated accuracy
    best_params = grid_search.best_params_
    best_cv_accuracy = grid_search.best_score_
    print(f'Best parameters found: {best_params}')
    print(f'Best cross-validation accuracy: {best_cv_accuracy:.2f}')
    # Train the final model with the best parameters on the entire dataset
    best_tree = grid_search.best_estimator_
    best_tree.fit(df, y)
    # Visualize the decision tree
    plt.figure(figsize=(60, 30))
    plot_tree(best_tree, filled=True, feature_names=df.columns, rounded=True, fontsize=28)
    plt.show()


def get_columns_w_names(df, strings):
    """
    Sina Bonakdar, March 2025
    Inputs:
    df: DataFrame containing all the features
    strings: A single string or a list of strings. Columns containing any of these strings will be retained.
    """
    import pandas as pd
    # Ensure strings is a list for consistent processing
    if isinstance(strings, str):
        strings = [strings]
    # Create a list of column names that contain any of the strings in the list
    colnames = [col for col in df.columns if any(s in col for s in strings)]
    DF = df[colnames]
    return DF


def calc_perc_variance(X,model):
    """
    Sina Bonakdar, March 2025
    Calculate percent of variance captured across LV1 and LV2 in PLSDA
    Will work in PLSDA_plots function 
    """
    import numpy as np
    
    # Compute total variance of X
    total_variance_X = np.var(X.values, axis=0).sum()
    # Compute variance for each component
    x_scores = model.x_scores_
    explained_variance_lv1 = np.var(x_scores[:, 0]) / total_variance_X
    explained_variance_lv2 = np.var(x_scores[:, 1]) / total_variance_X
    # Convert to percentage
    percent_variance_lv1 = explained_variance_lv1 * 100
    percent_variance_lv2 = explained_variance_lv2 * 100

    return percent_variance_lv1, percent_variance_lv2


def PLSDA_plots(xblock, yblock, num_components=None, orthogonalize=False, num_loadings=None, LV1=True, LV2=True):
    """
    Sina Bonakdar, April 2025
    This function generated for Elastic Net selected features but can be used on any data
    If number of features is small this function generates more beautiful plots, otherwise 
    use plsda_scores_loading function 

    Updated in April 2025: you can set the number of loadings for plotting by adjusting 
    num_loadings. e.g. num_loadings = 5, plot the top 5 positive and 5 negative loadings. 
    Also, updated in a way that user set which LV loadings want to plot. 
    If LV1=True: pls1 will plot, If LV2=True: pls 2 will plot. 
    """
    from sklearn.cross_decomposition import PLSRegression
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np 
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from pyopls import OPLS
    import matplotlib.gridspec as gridspec
    
    X = xblock
    feature_names = X.columns
    # Check if yblock is a numpy array
    if isinstance(yblock, np.ndarray):
        # Reshape yblock for OneHotEncoder
        y2 = yblock.reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        Y_encoded = encoder.fit_transform(y2)
        # Create a DataFrame with custom column names
        Y = pd.DataFrame(Y_encoded, columns=['Class1', 'Class2'])
    else:
        # If it's already a DataFrame, we assume it's processed
        Y = yblock
    
    if num_components is None:
        num_components = 2
        
    #Orthogonolized
    if orthogonalize==True:
        Y_single_column = np.argmax(Y.values, axis=1)  # Ensure using values for ndarray operation
        X = X.astype(float)
        opls = OPLS(n_components=2)
        X = opls.fit_transform(X, Y_single_column)
        X = pd.DataFrame(X, columns=feature_names)
        
    pls = PLSRegression(n_components=num_components)
    #Fit the model
    pls.fit(X, Y)
    # Extract the scores for the first two latent variables
    scores = pls.transform(X)
    # Extract loadings
    loadings = pls.x_loadings_
    
    def get_sorted_key_features(loadings_vector, num_loadings):
        if num_loadings is not None:
            top_indices = loadings_vector.argsort()[-num_loadings:][::-1]
            bottom_indices = loadings_vector.argsort()[:num_loadings]
            indices = np.concatenate((top_indices, bottom_indices))
            indices = indices[np.argsort(loadings_vector[indices])]  # Sort within the selected features
        else:
            indices = np.argsort(loadings_vector)
        
        return indices

    key_features_pls1 = get_sorted_key_features(loadings.T[0], num_loadings)
    key_features_pls2 = get_sorted_key_features(loadings.T[1], num_loadings)
    
    #Calculate the percent of variance captured
    perc_var_LV1, perc_var_LV2 = calc_perc_variance(X, pls)

    # Create color mapping
    color_mapping = {'BV': 'r', 'no BV': 'b'}

    # Set up a 2x2 plotting area
    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 3])  # Adjust the height ratios as needed

    # Plot (1, 1) - Scores plot with updated legend
    #sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=Y['Class2'], ax=axes[0, 0], palette=['b', 'r'])
    ax0 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=Y['Class2'].map({1.0: 'BV', 0.0: 'no BV'}), 
                    ax=ax0, palette=color_mapping, s=100)
    ax0.legend(title='', edgecolor='black', fontsize=16)
    #ax0.set_title('PLS-DA Scores Plot')
    ax0.set_xlabel(f'Scores on LV1 ({perc_var_LV1.round(2)}%)', fontsize=18)
    ax0.set_ylabel(f'Scores on LV2 ({perc_var_LV2.round(2)}%)', fontsize=18)
    ax0.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax0.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax0.tick_params(axis='both', direction='in')

    # Plot (1, 2) - Loadings plot for PLS2
    sorted_loadings2 = loadings.T[1, key_features_pls2]
    colors_pls2 = ['b' if val < 0 else 'r' for val in sorted_loadings2]
    if LV2:
        ax1 = fig.add_subplot(gs[0, 1])
        sorted_loadings2 = loadings.T[1, key_features_pls2]
        colors_pls2 = ['b' if val < 0 else 'r' for val in sorted_loadings2]
        sns.barplot(x=feature_names[key_features_pls2], y=sorted_loadings2, ax=ax1, palette=colors_pls2)
        ax1.set_xlabel('')
        ax1.set_ylabel(f'Loadings on LV2 ({perc_var_LV2.round(2)}%)')
        ax1.set_xticklabels(feature_names[key_features_pls2], rotation=45, ha='right')
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()
        ax1.tick_params(axis='both', direction='in')
    else:
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.axis('off')

    # Plot (2, 1) - Loadings plot for PLS1
    if LV1:
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_loadings1 = loadings.T[0, key_features_pls1]
        colors_pls1 = ['b' if val < 0 else 'r' for val in sorted_loadings1]
        sns.barplot(x=sorted_loadings1, y=feature_names[key_features_pls1], ax=ax2, palette=colors_pls1)
        ax2.set_xlabel(f'Loadings on LV1 ({perc_var_LV1.round(2)}%)', fontsize=18)
        ax2.set_ylabel('')
        ax2.tick_params(axis='both', direction='in', labelsize=20)
    else:
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
    
    # Hide the empty subplot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    #plt.suptitle('PLSDA Model', fontsize=24, fontweight='bold')
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)
    plt.show()



def create_amsel_label():
    """
    Sina Bonakdar, March 2025
    Using SBV and ABV values to create the label column
    if SBV or ABV==1: label=1, else: label=0
    original data file: '/Users/bonakdar/Desktop/Arnold Lab/QE/HMP/40168_2013_28_MOESM1_ESM.xlsx'
    """
    import pandas as pd 
    # Read the Excel file, initially without specifying headers
    df_clinic = pd.read_excel('/Users/bonakdar/Desktop/Arnold Lab/QE/HMP/40168_2013_28_MOESM1_ESM.xlsx', header=None)
    # Drop the first three rows
    df_clinic = df_clinic.drop([0, 1, 2])
    # Reset the index to prevent having gaps due to dropped rows
    df_clinic.reset_index(drop=True, inplace=True)
    # Use the third row as the column header
    df_clinic.columns = df_clinic.iloc[0]
    # Drop the now redundant third row
    df_clinic = df_clinic.drop(0)
    # Reset the index again to maintain a contiguous index
    df_clinic.reset_index(drop=True, inplace=True)
    
    # Define a function to rename the sampleID values: We should make the sampleID values format similar to the HMP_ALL_DATA data
    def rename_sample_id(sample_id):
        # Split the input string into its components
        parts = sample_id.split('.')
        # Extract the number after 's' and pad it to three digits
        subject_number = parts[0][1:].zfill(3)
        # Extract the week and day parts
        week_part = parts[1].replace('d', 'D')
        # Format the new sampleID string
        new_sample_id = f"UAB{subject_number}_{week_part.upper()}"
        return new_sample_id

    # Apply the function to the entire 'sampleID' column
    df_clinic['sampleID'] = df_clinic['sampleID'].apply(rename_sample_id)

    #create the label column: If SBV or ABV == 1: label = 1, else: label=0
    for idx in range(df_clinic.shape[0]):
        if df_clinic.loc[idx,'SBV']==1 or df_clinic.loc[idx,'ABV']==1:
            df_clinic.loc[idx,'label'] = 1
        else: 
            df_clinic.loc[idx,'label'] = 0

    # Rename the 'sampleID' column to 'UID'
    df_clinic = df_clinic.rename(columns={'sampleID': 'UID'})
    return df_clinic



def df_abundance_amsel(df_clinic,filename=None, log=False, rel_abund=False, zscore=False, cst=False):
    """
    Sina Bonakdar March, 2025
    Load data from a specified Excel file into a pandas DataFrame.
    Uses a default file path if no filename is provided.

    Parameters:
    filename (str): The path to the Excel file.
    df_clinic is the output from create_amsel_label() function 
    log=True: perform natural log transformation
    rel_abund=True: perform relative abundance data
    zscore=True: preprocess of relative abundance using zscore
    cst=True: Will record CST-I,II,III,IV,V as a column as well


    Returns:
    Cleaned dataframe with 1732 rows and 83 columns 
    including bacterial abundance + Amsel Class for each timepoint and sample. 
    """
    from sklearn import preprocessing
    import pandas as pd
    if not filename:
        filename = '/Users/bonakdar/Desktop/Arnold Lab/QE/HMP/HMP_ALL_DATA_Release2_with_METABOLOMICS_ALL_12172017.xlsx'
    df = pd.read_excel(filename)
    # Perform a left merge on the 'UID' column
    df = df.merge(df_clinic[['UID', 'label']], on='UID', how='left')
    #drop the samples with mensturation for all time points 
    values_to_drop = [4, 19, 63, 79]
    df = df[~df['PID'].isin(values_to_drop)]
    #drop all the columns except SERIAL, PID, menstruation and taxa and label
    cols_to_drop = list((0,1,3))
    cols_to_drop += list(range(5, 15))  
    if cst==True:
        cols_to_drop += list(range(16, 145))
    else:
        cols_to_drop += list(range(16, 146))
    cols_to_drop += list(range(df.shape[1] - 9, df.shape[1]-1))
    df.drop(df.columns[cols_to_drop], axis=1, inplace=True)
    #drop the columns with more than 3800 nan values 
    nan_counts = df.isna().sum()
    cols_to_drop = nan_counts[nan_counts >= 3800].index
    if cst==True:
        df.drop(columns=cols_to_drop[2:-1], inplace=True) #drop all columns except menstruation 
    else:
        df.drop(columns=cols_to_drop[1:-1], inplace=True)
    #drop rows with nan values and Specify columns where NaN values should be ignored
    ignore_columns = ['PID', 'MENSTRUATION', 'SERIAL']
    df = df.dropna(subset=[col for col in df.columns if col not in ignore_columns])
    df.reset_index(drop=True, inplace=True)
    # Define the mapping for the replacement in the label column
    label_mapping = {0: 'NO_BV', 1: 'BV'}
    df['label'] = df['label'].map(label_mapping)
    if cst==True:
        bacteria_columns = df.columns[4:-1]
    else:
        bacteria_columns = df.columns[3:-1]
    if log==True:
        #perform log transformation
        small_number = 1e-10
        df[bacteria_columns] = df[bacteria_columns].replace(0, small_number)
        df[bacteria_columns] = np.log(df[bacteria_columns])
        print('log transformed data')
        return df   
    if rel_abund==True:
        #create relative abundance data 
        row_sums = df[bacteria_columns].sum(axis=1) # Calculate the sum of the relevant columns for each row
        df[bacteria_columns] = df[bacteria_columns].div(row_sums, axis=0)
        if zscore==True:
        #preprocessing of df: standardize the data for each row. Makes each sample to have zero mean and 1 std. Known as zscore  normalization 
            for col in bacteria_columns:
                df[col] = preprocessing.scale(df[col].values)  # Scale the data to make it between 0-1.
            print('relative abundance + zscore data')
            return df
        else:
            print('relative abundance data')
            return df      
    else:
        print('either log, rel_abund, or zscore has to be True')
        pass 



def split_blocks_by_CST_most_frequent(X, y,equal_selection=False):
    """
    Sina Bonakdar, April 2025
    Separate xblocks and corresponding y values according to the most frequent CST in premenses data
    X and y are the outputs from drop_blocks function 
    Output of this function is 5 tuples (5 CSTs), each containing X and y values for each data
    equal_selection=True: If frequency for 2 CSTs is equal, it randomly selects one
    Otherwise, it drops the xblock and y which had equal frequency. 
    
    """
    import random
    import numpy as np
    import pandas as pd
    # Example of creating empty lists for each category
    X_1, X_2, X_3, X_4, X_5 = [], [], [], [], []
    y_1, y_2, y_3, y_4, y_5 = [], [], [], [], []
    for i, xblock in enumerate(X):
        # Get the value count for the 'CST_HL' column
        cst_counts = xblock['CST_HL'].value_counts()
        #print('cst counts: ',cst_counts)
        # Find the maximum frequency value(s)
        max_freq = cst_counts.max()
        #print('maximum freq: ', max_freq)
        candidates = cst_counts[cst_counts == max_freq].index.tolist()
        #print('candidates: ', candidates)

        #Initialize chosen_category to none 
        chosen_category = None
        if len(candidates) == 1: 
            chosen_category = candidates[0]
            #print('chosen (no tie): ', chosen_category)
        elif equal_selection==True:
            # Choose one randomly if there's a tie
            chosen_category = random.choice(candidates)
            #print('randomly chosen (tie): ',chosen_category)

        #Only proceed to append if a valid category was selected. 
        if chosen_category != None: 
            xblock = xblock.drop(columns=['CST_HL'])
         # Append to the corresponding list if chosen_category exist
            if chosen_category == 1:
                X_1.append(xblock)
                y_1.append(y[i])
            elif chosen_category == 2:
                X_2.append(xblock)
                y_2.append(y[i])
            elif chosen_category == 3:
                X_3.append(xblock)
                y_3.append(y[i])
            elif chosen_category == 4:
                X_4.append(xblock)
                y_4.append(y[i])
            elif chosen_category == 5:
                X_5.append(xblock)
                y_5.append(y[i])
    # convert to numpy arrays
    y_1, y_2, y_3, y_4, y_5 = map(np.array, [y_1, y_2, y_3, y_4, y_5])
    X_1 = np.array(X_1, dtype=object)
    X_2 = np.array(X_2, dtype=object)
    X_3 = np.array(X_3, dtype=object)
    X_4 = np.array(X_4, dtype=object)
    X_5 = np.array(X_5, dtype=object)
    print('length of X_1 and y_1: ',len(X_1), len(y_1))
    print('length of X_2 and y_2: ',len(X_2), len(y_2))
    print('length of X_3 and y_3: ',len(X_3), len(y_3))
    print('length of X_4 and y_4: ',len(X_4), len(y_4))
    print('length of X_5 and y_5: ',len(X_5), len(y_5))
    
    return (X_1, y_1), (X_2, y_2), (X_3, y_3), (X_4, y_4), (X_5, y_5)



def binary_label_selection(X,y,label1,label2):
    """
    Sina Bonakdar, March 2025
    In case of comparision between transition to BV/healthy
    from healthy/BV we have 4 labels. 
    We need to binarized them to perform our analysis. 
    np.unique(y) = [0,1,2,3]
    X and y are the inputs from drop_blocks function
    X: include all xblocks
    y: include all 4 labels
    val1: first label to keep
    val2: second label to keep
    Output: Final X and y with only 2 labels (label1=0, label2=1)
    """
    import numpy as np
    import pandas as pd
    lst=[]
    for idx,val in enumerate(y):
        if val==label1 or val==label2:
            lst.append(idx)
            
    X_selected = X[lst]
    y_selected= y[lst]
    y_selected[y_selected == label1] = 0
    y_selected[y_selected == label2] = 1
    return X_selected, y_selected



def metabolic_count(df=None, X=None):
    """
    Sina Bonakdar, March 2025
    Identify how many metabolic days measurement we have in: 
    df=df: in each sampleIDs
    X=X: in each block of X
    To run this function, df_abundance_nugent metabolic input has to be true to record the metabolic days. 
    """
    if df is not None:
        for pid, group in df.groupby('PID'):
            print('PID number {} has {} metabolic days'.format(pid, sum(group['Metabolomics'] == 1)))
    
    if X is not None:
        i = 0
        for xblock in X:
            print('block number {} has {} metabolic days'.format(i, sum(xblock['Metabolomics'] == 1)))
            i += 1




def evaluate_classifiers(X_train, X_test, y_train, y_test):
    """
    Sina Bonakdar, April 2025
    This function works with a specified set of train and test data. 
    For instance divide the data into 80:20 train:test ratio and run this model on it. 
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, mean_squared_error
    )
    import pandas as pd
    import numpy as np

    # Define classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(
            max_depth=None,
            max_features='log2',
            min_samples_leaf=1,
            min_samples_split=10,
            n_estimators=100,
            random_state=0
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
        "Linear SVM": LinearSVC(max_iter=10000, random_state=0),
        "RBF SVM": SVC(kernel='rbf', probability=True, random_state=0),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GBDT": GradientBoostingClassifier(n_estimators=100, random_state=0),
        "PLS-DA": PLSRegression(n_components=2),
        "Random Guess (Most Frequent)": DummyClassifier(strategy="most_frequent")
    }

    # Metrics to compute and store
    metrics_table = []

    for name, clf in classifiers.items():
        # Train and predict using each classifier
        clf.fit(X_train, y_train)

        if name == "PLS-DA":  # Special case for PLS-DA
            predictions = clf.predict(X_test)
            # Binarize at 0.5
            y_pred = (predictions >= 0.5).astype(int).flatten()
            y_proba = predictions.flatten()  # needed for AUC
        else:
            y_pred = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_test)[:, 1]
            else:
                y_proba = clf.decision_function(X_test)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

        # Metrics
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "Specificity": specificity,
            "MSE": mean_squared_error(y_test, y_pred),
        }

        if len(np.unique(y_test)) == 2:  # AUC only for binary
            metrics["AUC"] = roc_auc_score(y_test, y_proba)

        metrics_table.append(metrics)
    
    # Make DataFrame
    df_metrics = pd.DataFrame(metrics_table)
    return df_metrics




def Gajer_preprocessing_data(log=False, rel_abund=False, zscore=False):
    """
    Sina Bonakdar, April 2025
    preprocess the Gajer data with Menstruation 
    and make it similar format of HMP 
    Similar function as df_abundance_nugent
    """
    
    import warnings
    # Suppress all warnings globally
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    filename = '/Users/bonakdar/Desktop/Arnold Lab/QE/Gajer/Gajer_Supplement _menses_added_by_Sina.xlsx'
    df = pd.read_excel(filename, skiprows=3)  # Skip the first 3 rows initially
    df.columns = df.iloc[0]  # Set the new header from the first row
    df = df.drop(index=0).reset_index(drop=True)  # Drop the first row used for header and reset index
    # Determine the indices of columns to drop
    cols_to_drop = list((0,3,4,6,9,10))
    df.drop(df.columns[cols_to_drop], axis=1, inplace=True)
    df = df.dropna() #drop 48 rows with no Nugent Category 889 rows
    df.reset_index(drop=True, inplace=True) 

    df.rename(columns={'Subject ID': 'PID', 'Time in study': 'SERIAL', 
                       'Menstruation': 'MENSTRUATION', 'Nugent Categoryb': 'NUGENT_CLASS', 
                       'Community State Typec': 'CST_HL'}, inplace=True) #Make the column names similar to HMP (except bacteria)

    df['MENSTRUATION'] = df['MENSTRUATION'].replace(0, np.nan) #In HMP non menses days shown with nan. make it similar format
    df['NUGENT_CLASS'] = df['NUGENT_CLASS'].replace({'Low': 'NO_BV', 'Int': 'INTER_BV', 'High': 'BV'}) #same format of HMP
    df['CST_HL'] = df['CST_HL'].replace({'I': 1.0, 'II': 2.0, 'III': 3.0, 
                                         'IV-A': 4.0, 'IV-B': 4.0}) #same format of HMP
    df = df.sort_values(by=['PID', 'SERIAL'], ascending=[True, True])
    df.reset_index(drop=True, inplace=True) 

    # Filter out columns where the count of not zero or non nan values is less than 10
    non_zero_non_nan_counts = ((df != 0) & (~df.isna())).sum(axis=0)
    columns_to_keep = non_zero_non_nan_counts[non_zero_non_nan_counts >= 10].index
    df = df[columns_to_keep]

    bacteria_columns = df.columns[5:]

    if log==True:
        #perform log transformation
        small_number = 1e-10
        df[bacteria_columns] = df[bacteria_columns].replace(0, small_number)
        df[bacteria_columns] = np.log(df[bacteria_columns])
        print('log transformed data')
        return df
    if rel_abund==True:
        #create relative abundance data 
        row_sums = df[bacteria_columns].sum(axis=1) # Calculate the sum of the relevant columns for each row
        df[bacteria_columns] = df[bacteria_columns].div(row_sums, axis=0)
        if zscore==True:
        #preprocessing of df: standardize the data for each row. Makes each sample to have zero mean and 1 std. Known as zscore  normalization 
            for col in bacteria_columns:
                df[col] = preprocessing.scale(df[col].values)  # Scale the data to make it between 0-1.
            print('relative abundance + zscore data')
            return df
        else:
            print('relative abundance data')
            return df      
    else:
        print('either log, rel_abund, or zscore has to be True. This is original data')
        return df





def common_feature_selection_in_2_dataframes(df_hmp, df_gajer, rename_dict=None):
    """Sina Bonakdar, April 2025
    df1=hmp
    df2=Gajer
    Just use the common features and rename it according to rename_dict
    dictionary keys from df_gajer, and values from df_hmp
    """
    import pandas as pd

    if rename_dict is None:
        rename_dict = {'SERIAL': 'SERIAL', 'PID': 'PID', 'MENSTRUATION': 'MENSTRUATION', 'NUGENT_CLASS': 'NUGENT_CLASS', 
                       'L. iners': 'Lactobacillus_iners', 'L. crispatus': 'Lactobacillus_crispatus', 'L. jensenii': 'Lactobacillus_jensenii', 
                       'L. gasseri': 'Lactobacillus_gasseri', 'L.vaginalis': 'Lactobacillus_vaginalis','Parvimonas': 'Parvimonas_micra', 
                       'Enterococcus': 'Enterococcus_faecalis', 'Finegoldia': 'Finegoldia_magna', 'Gemella': 'Gemella', 
                       'Sneathia': 'Sneathia_sanguinegens', 'Atopobium': 'Atopobium_vaginae', 'Corynebacterium': 'Corynebacterium_accolens',
                       'Prevotella': 'Prevotella_bivia', 'Gardnerella': 'Gardnerella_vaginalis', 'Megasphaera': 'Megasphaera_sp._type_2', 
                       'Staphylococcus': 'Staphylococcus_hominis'}

    # Step 1: Keep only the columns that are keys in the dictionary
    columns_to_keep = list(rename_dict.keys())
    common_columns_gajer = df_gajer.columns.intersection(columns_to_keep)
    df_gajer_filtered = df_gajer[common_columns_gajer]
    # Rename the columns based on the dictionary mapping
    df_gajer_filtered = df_gajer_filtered.rename(columns=rename_dict)

    # Step 2: Align df_hmp to match the columns of df_gajer_filtered
    columns_to_keep_in_hmp = df_gajer_filtered.columns.intersection(df_hmp.columns)
    df_hmp_filtered = df_hmp[columns_to_keep_in_hmp]

    # Rearrange the columns of df_hmp_filtered to match the order of df_gajer_filtered
    df_hmp_filtered = df_hmp_filtered[df_gajer_filtered.columns]

    return df_hmp_filtered, df_gajer_filtered
    


def lstm_model_train_test(X_train, y_train, X_test, y_test, padding_value=None):
    """
    Sina Bonakdar, September 2025
    Modified LSTM model, handling the input from DataFrames, numpy arrays, or lists.
    Works with sequential input for LSTM and runs on GPU (MPS) if available.
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization, Masking
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix, mean_squared_error

    # --- Device setup ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("Using GPU:", gpus[0])
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. Using CPU.")

    # --- Helper to normalize sequence input ---
    def prepare_sequences(X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(X, np.ndarray):
            if X.ndim == 3:   # (samples, timesteps, features)
                return X
            elif X.ndim == 2: # (samples, timesteps) → add feature dim
                return X[:, :, np.newaxis]
            elif X.ndim == 1: # array of objects/lists
                return list(X)
            else:
                raise ValueError(f"Unexpected numpy array shape: {X.shape}")

        elif isinstance(X, list):  # already list of sequences
            return X

        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

    # Prepare train and test
    X_train = prepare_sequences(X_train)
    X_test = prepare_sequences(X_test)

    # --- Padding if lists ---
    if isinstance(X_train, list) or isinstance(X_test, list):
        max_length = max(max(len(seq) for seq in X_train), max(len(seq) for seq in X_test))
        n_features = X_train[0].shape[1] if hasattr(X_train[0], "shape") and X_train[0].ndim > 1 else 1
        X_train = pad_sequences(X_train, maxlen=max_length, dtype="float32", padding="post", value=padding_value)
        X_test = pad_sequences(X_test, maxlen=max_length, dtype="float32", padding="post", value=padding_value)
        if n_features == 1 and X_train.ndim == 2:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    else:
        # Already numpy arrays
        max_length = X_train.shape[1]
        n_features = X_train.shape[2]

    # --- Labels ---
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # --- Model ---
    model = Sequential()
    if padding_value is not None:
        model.add(Masking(mask_value=padding_value, input_shape=(max_length, n_features)))

    model.add(LSTM(units=512, activation="tanh", return_sequences=True, input_shape=(max_length, n_features)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(LSTM(units=512, activation="tanh", return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(LSTM(units=512, activation="tanh"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(64, activation="tanh"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))  # binary classification

    opt = tf.keras.optimizers.RMSprop(learning_rate=1e-3, decay=1e-5)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True)

    # --- Train ---
    model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # --- Predict ---
    y_pred_probs = model.predict(X_test)[:, 1]
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # --- Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    mse = mean_squared_error(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    print(f"Test Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Specificity: {specificity}")
    print(f"MSE: {mse}")
    print(f"ROC AUC: {roc_auc}")

    return model, accuracy, precision, recall, f1, specificity, mse, roc_auc





def df_ahrens_abundance_nugent(filename=None, log=False, rel_abund=False, zscore=False, cst=False, treatment=False): 
    """
    Sina Bonakdar Aug, 2025
    Load data from a specified Excel file into a pandas DataFrame (Ahrens et al dataframe).
    Uses a default file path if no filename is provided.

    Parameters:
    filename (str): The path to the Excel file.
    log=True: perform natural log transformation
    rel_abund=True: perform relative abundance data
    zscore=True: preprocess of relative abundance using zscore
    cst=True: Will record CST-I,II,III,IV,V as a column as well
    treatment=True: Will record the Cohort column which includes treatment info
    
    Returns:
    Cleaned dataframe with X rows and X columns 
    including bacterial data + NUGENT Class for each timepoint and sample. 
    """
    import pandas as pd 
    from sklearn import preprocessing
    import numpy as np
    if not filename:
        filename = '/Users/bonakdar/Desktop/Arnold Lab/Ahrens Dataset/pone.0236036.s002.xlsx'
    # Load your DataFrame
    df = pd.read_excel(filename)
    
    # Determine the indices of columns to drop
    cols_to_drop = list((0,2,6))
    if cst==True and treatment==True:
        #df = df.rename(columns={'Cohort': 'CST'})
        df['CST'] = df['Cohort'].str[:4]
        bacteria_columns = df.columns[8:-1]
    if cst==True and treatment==False:
        df = df.rename(columns={'Cohort': 'CST'})
        df['CST'] = df['CST'].str[:4]
        bacteria_columns = df.columns[8:]
    if cst==False and treatment==True: 
        bacteria_columns = df.columns[8:]
    if cst==False and treatment==False: 
        cols_to_drop += [4]
        bacteria_columns = df.columns[8:]
    df.drop(df.columns[cols_to_drop], axis=1, inplace=True)
    #Take the label column named (NUGENT_CLASS) to the end: 
    column_to_move = 'NUGENT_CLASS'
    columns = list(df.columns)
    columns.append(columns.pop(columns.index(column_to_move)))
    df = df[columns]
    #drop rows with nan values
    df = df.dropna().reset_index(drop=True)
    #Fix the menses columns
    df['MENSTRUATION'] = df['MENSTRUATION'].apply(lambda x: 1 if x==1 else np.nan)

    if log==True:
        #perform log transformation
        small_number = 1e-10
        df[bacteria_columns] = df[bacteria_columns].replace(0, small_number)
        df[bacteria_columns] = np.log(df[bacteria_columns])
        print('log transformed data')
        return df   
    if rel_abund==True:
        #create relative abundance data 
        row_sums = df[bacteria_columns].sum(axis=1) # Calculate the sum of the relevant columns for each row
        df[bacteria_columns] = df[bacteria_columns].div(row_sums, axis=0)
        if zscore==True:
        #preprocessing of df: standardize the data for each row. Makes each sample to have zero mean and 1 std. Known as zscore  normalization 
            for col in bacteria_columns:
                df[col] = preprocessing.scale(df[col].values)  # Scale the data to make it between 0-1.
            print('relative abundance + zscore data')
            return df
        else:
            print('relative abundance data')
            return df      
    else:
        print('either log, rel_abund, or zscore has to be True. This is original data')
        return df



def pre_mens_identify_based_on_menses(df_p, df_m, y_m):
    """
    Sina Bonakdar, August 2025
    
    Identify premenses blocks for each block of menses and
    drop the menses blocks with no corresponding pre-menses block.
    
    Parameters
    ----------
    df_p : list of pd.DataFrame
        List of premenses blocks.
    df_m : list or np.ndarray of pd.DataFrame
        List of menses blocks.
    y_m : list or np.ndarray
        Labels corresponding to menses blocks.
    
    Returns
    -------
    premenses_blocks : list of pd.DataFrame
        Matching premenses blocks.
    df_m_updated : np.ndarray
        Updated menses blocks with missing ones removed.
    y_m_updated : np.ndarray
        Updated labels corresponding to the updated menses blocks.
    """
    
    import numpy as np
    import pandas as pd

    def get_block_indices(block):
        """Return the index numbers for a DataFrame block."""
        return block.index.to_list()
    
    premenses_blocks = []
    missing_indices = []

    for i, m_block in enumerate(df_m):
        m_indices = get_block_indices(m_block)
        premenses_end = m_indices[0] - 1
        #print(premenses_end)
        # Try to find a premenses block ending at premenses_end
        found = False
        for p_block in df_p:
            p_indices = get_block_indices(p_block)
            if p_indices[-1] == premenses_end:
                premenses_blocks.append(p_block)
                found = True
                break
        if not found:
            print(f"Premenses end index {premenses_end} for menses block {i} "
                  f"does NOT exist in df_p and will be deleted")
            missing_indices.append(i)
    # Remove missing menses blocks
    df_m_updated = np.delete(df_m, missing_indices, axis=0)
    y_m_updated = np.delete(y_m, missing_indices, axis=0)
    
    return premenses_blocks, df_m_updated, y_m_updated





def classify_pre_menses_blocks(premenses_blocks, df_m, y_m,
                               X_1_p, X_2_p, X_3_p, X_4_p, X_5_p):
    """
    Classify menses blocks into pre-healthy, pre-intermediate, and pre-dysbiotic categories
    based on their corresponding premenses CST label.
    
    Parameters
    ----------
    premenses_blocks : list of pd.DataFrame
        List of premenses blocks from pre_mens_identify_based_on_menses().
    df_m : list or np.ndarray of pd.DataFrame
        Updated menses blocks.
    y_m : list or np.ndarray
        Updated menses block labels.
    X_1_p ... X_5_p : list of pd.DataFrame
        Reference premenses blocks for CST1 through CST5.
    
    Returns
    -------
    X_pre_healthy, y_pre_healthy,
    X_pre_inter, y_pre_inter,
    X_pre_dysbio, y_pre_dysbio : list
        Lists of DataFrames and labels for each category (only if they exist).
    """
    import numpy as np
    import pandas as pd
    
    X_pre_healthy, y_pre_healthy = [], []
    X_pre_inter, y_pre_inter = [], []
    X_pre_dysbio, y_pre_dysbio = [], []
    
    labels = ['CST1', 'CST2', 'CST3', 'CST4', 'CST5']
    
    for i, m_block in enumerate(df_m):
        # Remove CST_HL column from premenses block before matching
        target_df = premenses_blocks[i].drop(columns=['CST_HL'], errors='ignore')
        
        found_flags = [
            any(target_df.equals(df) for df in X_1_p),
            any(target_df.equals(df) for df in X_2_p),
            any(target_df.equals(df) for df in X_3_p),
            any(target_df.equals(df) for df in X_4_p),
            any(target_df.equals(df) for df in X_5_p)
        ]
        
        # Determine CST label
        block_label = None
        for found, label in zip(found_flags, labels):
            if found:
                block_label = label
                break
        
        # Categorize
        if block_label in ['CST1', 'CST2', 'CST5']:
            X_pre_healthy.append(m_block)
            y_pre_healthy.append(y_m[i])
        elif block_label == 'CST3':
            X_pre_inter.append(m_block)
            y_pre_inter.append(y_m[i])
        elif block_label == 'CST4':
            X_pre_dysbio.append(m_block)
            y_pre_dysbio.append(y_m[i])

    print("Pre-healthy blocks array length:", len(X_pre_healthy), len(y_pre_healthy))
    print("Pre-inter blocks array length:", len(X_pre_inter), len(y_pre_inter))
    print("Pre-dysbiotic blocks array length:", len(X_pre_dysbio), len(y_pre_dysbio))
    
    return (X_pre_healthy, y_pre_healthy,
            X_pre_inter, y_pre_inter,
            X_pre_dysbio, y_pre_dysbio)



def plot_staph_dotplot(df_avg, df_type='rel_abund'):
    """
    Sina Bonakdar, August 2025 
    Dot plots of all staph species in Ahrens et al dataset
    df_avg: is the input dataset after grouped and averaged. 
    Dot plot of Staph species abundances across cohorts with mean ± SEM.

    Parameters:
        df_avg (pd.DataFrame): DataFrame containing 'Cohort' column and Staph species columns.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    staph_cols = [col for col in df_avg.columns if 'Staph' in col]
    
    if not staph_cols:
        print("No Staph columns found in the DataFrame.")
        return

    n_plots = len(staph_cols)
    n_cols = 4
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 10*n_rows))
    axes = axes.flatten()
    
    sns.set_style("white")
    
    for idx, staph_col in enumerate(staph_cols):
        ax = axes[idx]
        labels = df_avg['Cohort'].unique()
        
        for i, label in enumerate(labels):
            group = df_avg.loc[df_avg['Cohort'] == label, staph_col]
            n = len(group)
            x = np.random.normal(i, 0.05, size=n)  # jitter around x=i
            
            # plot individual points
            ax.plot(x, group, 'o', markersize=15, alpha=0.7)
            
            # mean and SEM
            mean = np.mean(group)
            sem = np.std(group, ddof=1) / np.sqrt(n)
            ax.errorbar(i, mean, yerr=sem, fmt='_', color='black', linewidth=4, capsize=10)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=22)
        ax.set_title(staph_col, fontsize=30)
        ax.set_ylabel("Mean abundance", fontsize=22)
        ax.set_ylim(df_avg[staph_col].min() - 0.01, df_avg[staph_col].max() + 0.02)
        ax.tick_params(axis='y', labelsize=22) 
        ax.set_ylim(df_avg[staph_col].min() - 1, df_avg[staph_col].max() + 1)
        if df_type=='rel_abund':
            ax.set_ylim(-0.01, 0.03)
        ax.set_facecolor('white')
        ax.grid(False)
    
    # remove empty axes
    for j in range(len(staph_cols), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()



def anova_kruskal(df):
    """"
    Sina Bonakdar, August 2025
    Perform statistical tests in columns starting with Staph
    statistical test is between different treatment groups under 'Cohort' column
    """

    import pandas as pd
    import pingouin as pg

    # We need to drop the columns when all values are zero as ANOVA doesn't work 
    df= df.loc[:, (df != 0).any(axis=0)]

    # Identify all Staphylococcus-related columns
    staph_cols = [col for col in df.columns if col.startswith("Staphylococcus")]

    # Prepare results storage
    results = []

    for col in staph_cols:
        # ANOVA
        aov = pg.anova(dv=col, between='Cohort', data=df)
        aov_res = aov.iloc[0][['F', 'p-unc']].rename({'F': 'F_stat', 'p-unc': 'anova_p'})

        # Kruskal–Wallis
        kw = pg.kruskal(data=df, dv=col, between='Cohort')
        kw_res = kw.loc['Kruskal', ['H', 'p-unc']].rename({'H': 'H_stat', 'p-unc': 'kruskal_p'})

        # Store in a single row
        combined = pd.Series({'Bacteria': col, **aov_res.to_dict(), **kw_res.to_dict()})
        results.append(combined)

    # Combine into DataFrame
    results_df = pd.DataFrame(results)
    results_df[['F_stat', 'anova_p', 'H_stat', 'kruskal_p']] = results_df[['F_stat', 'anova_p', 'H_stat', 'kruskal_p']].round(2)

    return results_df



def pca_2d_staph_ahrens(df):
    """
    Sina Bonakdar, August 2025
    Plot 2D PCA plot on staph species 
    colored by their treatment cohort 
    """
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Assuming df_avg is your DataFrame
    df= df.loc[:, (df != 0).any(axis=0)]
    
    # 1. Select only Staphylococcus species columns
    staph_cols = [col for col in df.columns if 'Staphylococcus' in col]
    X = df[staph_cols]
    
    # 2. Standardize the features (recommended for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    #X_scaled = X
    
    # 3. Apply PCA (2 components for 2D plot)
    pca = PCA(n_components=2)
    pca_scores = pca.fit_transform(X_scaled)
    
    # 4. Create a DataFrame with PCA results and Cohort info
    pca_df = pd.DataFrame(pca_scores, columns=['PC1', 'PC2'])
    pca_df['Cohort'] = df['Cohort']
    
    # 5. Plot PCA scores
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cohort', palette='tab10', s=100, alpha=0.8)
    plt.title('PCA of Staphylococcus Species')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.legend(title='Cohort')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def pca_3d_staph_ahrens(df):
    """
    Sina Bonakdar, August 2025
    Plot 3D PCA plot on staph species 
    colored by their treatment cohort 
    """
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import plotly.express as px
    
    # Assuming df_avg is your DataFrame
    df= df.loc[:, (df != 0).any(axis=0)]
    
    # 1. Select Staphylococcus species columns
    staph_cols = [col for col in df.columns if 'Staphylococcus' in col]
    X = df[staph_cols]
    
    # 2. Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    #X_scaled = X
    
    # 3. PCA with 3 components
    pca = PCA(n_components=3)
    pca_scores = pca.fit_transform(X_scaled)
    
    # 4. Create a DataFrame for plotting
    pca_df = pd.DataFrame(pca_scores, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Cohort'] = df['Cohort']
    
    # 5. Interactive 3D scatter plot with Plotly
    fig = px.scatter_3d(
        pca_df, 
        x='PC1', y='PC2', z='PC3',
        color='Cohort',
        hover_data=['Cohort'],
        #title='3D PCA of Staphylococcus Species'
    )
    
    # Show explained variance in axes labels and make figure larger
    fig.update_layout(
        width=600,  # increase width
        height=800,  # increase height
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var)'
        )
    )
    
    fig.show()



def PLSDA_ahrens_staph(df):
    """
    Sina Bonakdar, August 2025
    PLSDA of Ahrens dataset for all staph colored by cohort
    """

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    # Assuming df_avg is your DataFrame
    df= df.loc[:, (df != 0).any(axis=0)]
    
    # Assuming df_avg is already loaded as before (see previous code)
    species_cols = ['Staphylococcus epidermidis', 'Staphylococcus aureus', 'Staphylococcus unclassified',
                    'Staphylococcus haemolyticus', 'Staphylococcus hominis', 'Staphylococcus simulans',
                    #'Staphylococcus equorum', 'Staphylococcus intermedius', 'Staphylococcus caprae',
                    'Staphylococcus warneri']
    
    X = df[species_cols].values
    y = df['Cohort'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y).reshape(-1, 1) # Must be 2D array for OPLS
    
    from pyopls import OPLS
    
    # n_ortho=1 to compute one orthogonal component (OPLS-DA)
    opls = OPLS(n_components=2)
    
    X = opls.fit_transform(X, y_encoded)
    X = pd.DataFrame(X, columns=species_cols)
    
    pls = PLSRegression(n_components=2)
    #Fit the model
    pls.fit(X, y_encoded)
    # Extract the scores for the first two latent variables
    scores = pls.transform(X)
    # Extract loadings
    loadings = pls.x_loadings_
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    sns.set(style="white", font_scale=1.15)
    
    # 1. PLS-DA Scores Plot
    plt.figure(figsize=(8,6))
    for idx, cohort in enumerate(le.classes_):
        ix = np.where(y == cohort)
        plt.scatter(scores[ix, 0], scores[ix, 1], label=cohort, s=60, alpha=0.75)
    plt.xlabel('PLS1')
    plt.ylabel('PLS2')
    plt.title('PLS-DA Scores Plot (Staph Species by Cohort)')
    plt.legend(title='Cohort')
    plt.tight_layout()
    sns.despine()
    plt.show()
    
    # 2. LV1 Loadings Plot
    plt.figure(figsize=(7,5))
    sns.barplot(y=species_cols, x=loadings[:, 0], palette='Blues_r')
    plt.title('PLS-DA Loadings for LV1')
    plt.xlabel('Loading Value')
    plt.ylabel('Staph Species')
    plt.tight_layout()
    sns.despine()
    plt.show()



def ttest_mannwithney(df):
    """
    Sina Bonakdar, August 2025. Performs 2samples test and MW tests on Ahrens dataset 
    on staph species. compare each treatment with control group. 
    control here is 'ol'. 
    """
    import pandas as pd
    import scipy.stats as stats
    from statsmodels.stats.multitest import multipletests
    
  
    staph_cols = [col for col in df.columns if col.startswith("Staphylococcus")]
    
    results = []
    
    for species in staph_cols:
        control_vals = df.loc[df["Cohort"] == "ol", species]
        
        for cohort in df["Cohort"].unique():
            if cohort == "ol":
                continue
            
            treat_vals = df.loc[df["Cohort"] == cohort, species]
            
            # skip if fewer than 2 values in either group
            if len(treat_vals) < 2 or len(control_vals) < 2:
                t_stat, t_p = float("nan"), float("nan")
                mw_stat, mw_p = float("nan"), float("nan")
            else:
                # t-test (Welch’s)
                t_stat, t_p = stats.ttest_ind(treat_vals, control_vals, equal_var=False)
                
                # Mann-Whitney
                try:
                    mw_stat, mw_p = stats.mannwhitneyu(
                        treat_vals, control_vals, alternative="two-sided"
                    )
                except ValueError:  # identical distributions
                    mw_stat, mw_p = float("nan"), float("nan")
            
            results.append({
                "Species": species,
                "Treatment": cohort,
                "t_stat": t_stat,
                "t_pval": t_p,
                "MW_stat": mw_stat,
                "MW_pval": mw_p
            })
    
    results_df = pd.DataFrame(results)
    
    # ---- FDR correction safely ----
    def fdr_safe(pvals):
        mask = ~pd.isna(pvals)
        adj = pd.Series([float("nan")] * len(pvals))
        if mask.sum() > 0:
            adj_vals = multipletests(pvals[mask], method="fdr_bh")[1]
            adj[mask] = adj_vals
        return adj
    
    results_df["t_pval_adj"] = fdr_safe(results_df["t_pval"])
    results_df["MW_pval_adj"] = fdr_safe(results_df["MW_pval"])
    
    results_df = results_df[results_df["Treatment"].isin(["Azi", "Azi-MNZ", "Tetra"])]
    results_df = results_df.round(5)
    return results_df



def plot_staph_dotplot_hmp(df_avg, y_labels, df_type='rel_abund'):
    """
    Sina Bonakdar, August 2025
    Dot plots of all staph species in HMP dataset.
    Each plot shows bacterial abundances across BV (1) vs No BV (0) groups.

    Parameters:
        df_avg (pd.DataFrame): DataFrame containing Staph species abundances.
        y_labels (array-like): Labels for BV outcome (0 = no BV, 1 = BV).
        df_type (str): Plot type ('rel_abund' adjusts ylim to small values).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Copy df_avg and add labels
    df_plot = df_avg.copy()
    df_plot["BV_label"] = y_labels
    
    staph_cols = [col for col in df_plot.columns if 'Staphylococcus' in col]
    if not staph_cols:
        print("No Staph columns found in the DataFrame.")
        return

    n_plots = len(staph_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    axes = axes.flatten()
    
    sns.set_style("white")
    
    for idx, staph_col in enumerate(staph_cols):
        ax = axes[idx]
        
        for i, label in enumerate([0, 1]):  # 0 = No BV, 1 = BV
            group = df_plot.loc[df_plot["BV_label"] == label, staph_col]
            n = len(group)
            x = np.random.normal(i, 0.05, size=n)  # jitter
            
            # Scatter points
            ax.plot(x, group, 'o', markersize=8, alpha=0.6)
            
            # Mean ± SEM
            mean = np.mean(group)
            sem = np.std(group, ddof=1) / np.sqrt(n) if n > 1 else 0
            ax.errorbar(i, mean, yerr=sem, fmt='_', color='black', 
                        linewidth=2, capsize=6)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No BV (0)", "BV (1)"], fontsize=14)
        ax.set_title(staph_col, fontsize=18)
        ax.set_ylabel("Abundance", fontsize=14)
        
        if df_type == 'rel_abund':
            ax.set_ylim(-0.01, df_plot[staph_col].max() * 1.1)
        else:
            ax.set_ylim(df_plot[staph_col].min() - 1, df_plot[staph_col].max() + 1)
        
        ax.set_facecolor('white')
        ax.grid(False)
    
    # Remove empty axes
    for j in range(len(staph_cols), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()



def ttest_mannwhitney_hmp(y, X):
    """
    Sina Bonakdar, August 2025.
    Performs Welch's t-test and Mann-Whitney U test for each Staphylococcus species
    comparing BV (1) vs. no-BV (0), with FDR correction (Benjamini-Hochberg).
    
    Parameters
    ----------
    y : array-like
        Binary labels (1 = BV, 0 = no BV).
    X : pd.DataFrame
        Features (Staphylococcus species abundances).
    
    Returns
    -------
    pd.DataFrame with t-test, Mann-Whitney results, and FDR-adjusted p-values.
    """
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    from statsmodels.stats.multitest import multipletests
    
    y = np.array(y)
    results = []
    staph_cols = [col for col in X.columns if col.startswith("Staphylococcus")]
    
    for species in staph_cols:
        group1 = X.loc[y == 1, species].dropna()  # BV
        group0 = X.loc[y == 0, species].dropna()  # no BV
        
        if len(group1) < 2 or len(group0) < 2:
            t_stat, t_p = float("nan"), float("nan")
            mw_stat, mw_p = float("nan"), float("nan")
        else:
            t_stat, t_p = stats.ttest_ind(group1, group0, equal_var=False)
            try:
                mw_stat, mw_p = stats.mannwhitneyu(
                    group1, group0, alternative="two-sided"
                )
            except ValueError:
                mw_stat, mw_p = float("nan"), float("nan")
        
        results.append({
            "Species": species,
            "t_stat": t_stat,
            "t_pval": t_p,
            "MW_stat": mw_stat,
            "MW_pval": mw_p
        })
    
    df_res = pd.DataFrame(results)
    
    # FDR correction
    df_res['t_pval_adj'] = multipletests(df_res['t_pval'], method='fdr_bh')[1]
    df_res['MW_pval_adj'] = multipletests(df_res['MW_pval'], method='fdr_bh')[1]
    
    return df_res



def grid_search_classifiers(X_train, y_train, cv=5, scoring="f1"):
    """
    Sina Bonakdar, Aug 2025. 
    Runs grid search for tunable classifiers and returns best models + params.
    Also adds fixed models (PLS-DA, DummyClassifier) without tuning.
    Parameter optimization on the train data. 
    The output of the model is best_models which will be used in evaluate_classifiers_with_best_models 
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_squared_error)
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.dummy import DummyClassifier
    import pandas as pd
    import numpy as np
    
    # Ensure numpy arrays
    if hasattr(X_train, "values"):
        X_train = X_train.values
    if hasattr(y_train, "values"):
        y_train = y_train.values
    y_train = np.ravel(y_train)



    # Define classifiers + param grids
    param_grid = {
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [10, 50, 100, 200],
                "max_depth": [None, 3, 5, 10, 20],
                "min_samples_split": [2, 5, 7, 10],
                "max_features": ["sqrt", "log2"]
            }
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                "C": [0.01, 0.1, 1],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"]
            }
        ),
        "Linear SVM": (
            LinearSVC(max_iter=10000, random_state=42),
            {
                "C": [0.01, 0.1, 1]
            }
        ),
        "RBF SVM": (
            SVC(kernel="rbf", probability=True, random_state=42),
            {
                "C": [0.1, 0.1, 1],
                "gamma": ["scale", "auto"]
            }
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]
            }
        ),
        "GBDT": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [10, 50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [2,3, 5, 7, 10]
            }
        )
    }

    best_models = {}
    
    # Grid search for tunable models
    for name, (clf, grid) in param_grid.items():
        print(f"Running GridSearchCV for {name}...")
        gs = GridSearchCV(clf, grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        best_models[name] = {
            "best_estimator": gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_score": gs.best_score_
        }
        print(f"✅ {name}: Best Params = {gs.best_params_}, Best CV Score = {gs.best_score_:.4f}")

    # Add PLS-DA (fixed, no tuning)
    best_models["PLS-DA"] = {
        "best_estimator": PLSRegression(n_components=2),
        "best_params": {"n_components": 2},
        "best_score": None
    }

    # Add Dummy classifier (baseline)
    best_models["Random Guess (Most Frequent)"] = {
        "best_estimator": DummyClassifier(strategy="most_frequent"),
        "best_params": {"strategy": "most_frequent"},
        "best_score": None
    }

    return best_models


def evaluate_classifiers_with_best_models(best_models, X_train, y_train, X_test, y_test):
    
    """
    Sina Bonakdar, August 2025
    Use the best_models (models with optimized hyperparameters) to evaluate the prediction on the test set. 
    Evaluates classifiers (tuned ones + PLS-DA + Dummy) on test data.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_squared_error)
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.dummy import DummyClassifier
    import pandas as pd
    import numpy as np
    metrics_table = []

    for name, model_info in best_models.items():
        clf = model_info["best_estimator"]

        # Fit on training data
        clf.fit(X_train, y_train)

        if name == "PLS-DA":
            preds = clf.predict(X_test)
            y_pred = (preds >= 0.5).astype(int).flatten()
            y_proba = preds.flatten()
        else:
            y_pred = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_proba = clf.decision_function(X_test)
                # scale to [0,1]
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            else:
                y_proba = None

        # Confusion matrix & metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "Specificity": specificity,
            "MSE": mean_squared_error(y_test, y_pred),
            "Best Params": model_info["best_params"]
        }
        if y_proba is not None and len(np.unique(y_test)) == 2:
            metrics["AUC"] = roc_auc_score(y_test, y_proba)

        metrics_table.append(metrics)

    return pd.DataFrame(metrics_table)



import pandas as pd
import numpy as np
from sklearn.utils import resample

def f_upsampling(df_train, y_train, random_state=42):
    """
    Sina Bonakdar, September 2025. 
    Perform upsampling on the training set to balance minority and majority classes.
    Test set is kept unchanged.

    Parameters:
    -----------
    df_train : pd.DataFrame
        Training predictors
    y_train : np.ndarray
        Training labels (binary: 0 majority, 1 minority)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X_train_balanced : pd.DataFrame
        Upsampled training predictors
    y_train_balanced : np.ndarray
        Upsampled training labels
    X_test : pd.DataFrame
        Test predictors (unchanged)
    y_test : np.ndarray
        Test labels (unchanged)
    """

    # Combine predictors and labels
    train_data = pd.concat([df_train, pd.Series(y_train, name="label")], axis=1)

    # Separate majority and minority classes
    majority = train_data[train_data.label == 0]
    minority = train_data[train_data.label == 1]

    # Upsample minority class
    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=random_state)

    # Combine majority and upsampled minority
    train_upsampled = pd.concat([majority, minority_upsampled])

    # Shuffle dataset
    train_upsampled = train_upsampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split back into predictors (X) and labels (y)
    X_train_balanced = train_upsampled.drop("label", axis=1)
    y_train_balanced = train_upsampled["label"].values


    print("Before upsampling:", np.bincount(y_train))
    print("After upsampling:", np.bincount(y_train_balanced))

    return X_train_balanced, y_train_balanced



def lstm_shap_beeswarm(model, X_train, X_test, feature_names=None, padding_value=None):
    """
    Sina Bonakdar, Sep 2025
    Compute SHAP values and plot beeswarm for an LSTM model.
    
    Args:
        model: trained Keras LSTM model.
        X_train: training set (numpy or DataFrame).
        X_test: test set (numpy or DataFrame).
        feature_names: list of original feature names.
        padding_value: if sequences are padded, must match training.
        sample_size: number of background samples for SHAP.
    """
    import shap
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Ensure arrays
    if isinstance(X_train, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X_train.columns)
        X_train = np.expand_dims(X_train.to_numpy(), axis=1)
    if isinstance(X_test, pd.DataFrame):
        X_test = np.expand_dims(X_test.to_numpy(), axis=1)

    # Pad if needed
    max_length = max(len(seq) for seq in X_train)
    if padding_value is not None:
        X_train = pad_sequences(X_train, maxlen=max_length, dtype='float32', padding='post', value=padding_value)
        X_test = pad_sequences(X_test, maxlen=max_length, dtype='float32', padding='post', value=padding_value)

    # Flatten to 2D for SHAP
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    # If 1 timestep, keep original feature names
    if X_train.shape[1] == 1 and feature_names is not None:
        flat_feature_names = feature_names
    else:
        # Create names like feature_timestep
        n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
        flat_feature_names = [f"{feature_names[j]}_t{t}" 
                              if feature_names is not None else f"f{j}_t{t}" 
                              for t in range(n_timesteps) 
                              for j in range(n_features)]

    # Define prediction function
    def predict_fn(x):
        x_reshaped = x.reshape(x.shape[0], X_train.shape[1], X_train.shape[2])
        return model.predict(x_reshaped, verbose=0)

    # Background
    #background = shap.sample(X_train_flat, sample_size, random_state=42)

    # Explainer
    explainer = shap.KernelExplainer(predict_fn, X_train_flat)
    shap_values = explainer.shap_values(X_test_flat)

    # Beeswarm
    #plt.figure(figsize=(10, 6))
    #shap.summary_plot(shap_values[:,:,0], X_test_flat, feature_names=flat_feature_names, plot_type="dot", show=True)



        # Create summary plot without immediately showing it
    import matplotlib.pyplot as plt
    fig = shap.summary_plot(
        shap_values[:,:,0],  # for class 1
        X_test_flat,
        feature_names=flat_feature_names,
        plot_type='dot',
        max_display=20,
        show=False
    )
    
    # Get the current axes and change the font size of y-axis (feature names)
    plt.gca().tick_params(axis='y', labelsize=20)  # change 14 to your desired size
    
    # Show the plot
    plt.show()

    return shap_values


