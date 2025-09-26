import warnings
# Suppress all warnings globally
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd

def read_files(directory):
    """
    Sina Bonakdar, March 2025
    Set the directory is important as input
    Read xblock,yblock,patients,xnames which needed for elastic net 
    and plsda
    output is the 4 files mentioned above
    """
    import pandas as pd
    import numpy as np
    xblock = pd.read_csv('{}/xblock.csv'.format(directory))
    xblock.drop(columns=['Unnamed: 0'],inplace=True)
    yblock = pd.read_csv('{}/yblock.csv'.format(directory))
    yblock.drop(columns=['Unnamed: 0'],inplace=True)
    xnames = pd.read_csv('{}/xnames.csv'.format(directory))
    xnames.drop(columns=['Unnamed: 0'],inplace=True)
    patients = pd.read_csv('{}/patients.csv'.format(directory))
    patients.drop(columns=['Unnamed: 0'],inplace=True)
    return xblock, yblock, xnames, patients


def resample_creation(xblock, yblock, num_runs=None, r=0.9): 
    """
    Sina Bonakdar, April 2025
    #create the resample datasets for feeding to Elastic Network
    r: resampling based on 90% of smallest class but can be changed
    by default it creates 2000 subsamples unless specified 
    """
    if num_runs==None:
        num_runs=2000
    #Separation of disease from cntrl data 
    disease_idx = yblock.iloc[:, 1] == 1
    cntrl_idx = yblock.iloc[:, 0] == 1
    disease_data = xblock.loc[disease_idx,:]
    cntrl_data = xblock.loc[cntrl_idx,:]
    #select class size for resampling
    ssize = np.floor(r * min(sum(disease_idx),sum(cntrl_idx)))
    
    #Generate resampled dataset
    from sklearn.utils import resample
    seed_val = np.arange(1, num_runs + 1)  # Create an array from 1 to num_runs=2000
    # Initialize empty lists to store resampled data
    all_cntrl_data = []
    all_disease_data = []
    # Populate resampled datasets
    for seed in seed_val:
        # Set the random seed for reproducibility
        np.random.seed(seed) 
        # Generate random sample from disease_data
        disease_x = resample(disease_data, n_samples=int(ssize), replace=True, random_state=seed)
        # Generate random sample from cntrl_data
        cntrl_x = resample(cntrl_data, n_samples=int(ssize), replace=True, random_state=seed)
        # Append the resampled data to the respective list
        all_disease_data.append(disease_x)
        all_cntrl_data.append(cntrl_x)
    # convert lists to a 3D NumPy array if needed. we convert the lists of DataFrames to NumPy arrays
    all_disease_data_np = np.array([df.to_numpy() for df in all_disease_data])
    all_cntrl_data_np = np.array([df.to_numpy() for df in all_cntrl_data])
    # At this point, all_disease_data_np and all_cntrl_data_np hold your resampled datasets as NumPy arrays.
    return all_disease_data_np, all_cntrl_data_np



def elastic_network(all_disease_data, all_cntrl_data, num_runs=None, ssize=None, kfold_val=None, l1_ratio=None, n_jobs=None):
    """
    Sina Bonakdar, March 2025
    Run CV elastic network model and get the coefficients according to the most optimized lambda (lowest CV error)
    Define hyperparameters for Elastic Network 
    alpha in ElasticNet in python corresponds to the lambda parameter in glmnet: number of lambdas to try
    This function should take several hours to run 
    """
    if num_runs==None: #total number of sample subsets and iterations
        num_runs=2000
    if ssize==None: #0.9*size of smallest class
        ssize=all_disease_data.shape[1]
    if kfold_val==None: #define the # of folds for tuning hyperparameter lambda in glmnet (or alpha in python)
        kfold_val=5
    if l1_ratio==None: #l1_ratio is the same as alpha in glmnet. We set to have equal weights between L1 and L2 norms. 
        l1_ratio=0.5
    if n_jobs==None: #Number of cores to use, -1: all cores
        n_jobs=-1

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import ElasticNetCV
    # Initialize an array to store optimized coefficients
    all_B_min = np.zeros((num_runs, all_disease_data.shape[2]))

    # Run ElasticNetCV on each resampled dataset
    for set_id in range(num_runs):
        print('run number: {}'.format(str(set_id)))
        # Extract disease and control data
        ex_disease_data = all_disease_data[set_id]  # Assuming class 1 is disease
        ex_cntrl_data = all_cntrl_data[set_id]  # Assuming class 0 is control

        # Combine data to form a X matrix and corresponding y labels
        combine_x = np.vstack((ex_disease_data, ex_cntrl_data))
        combine_y = np.concatenate((np.ones(ssize), np.zeros(ssize)))

        # Fit ElasticNetCV to find the optimal coefficients
        enet_cv = ElasticNetCV(cv=kfold_val, l1_ratio=l1_ratio,n_alphas=100, max_iter=1000,tol=0.0000001, n_jobs=n_jobs)
        enet_cv.fit(combine_x, combine_y)

        # Retrieve the coefficients for the best model
        all_B_min[set_id] = enet_cv.coef_
        

    # At this point, all_B_min contains the optimized coefficients for each run
    return all_B_min



def freq_of_selection(all_B_min,xnames): 
    """
    Sina Bonakdar, March 2025
    Create a list of all features with the frequency of selection for each
    sorted in descending order
    """
    num_runs = all_B_min.shape[0]
    fr_sel_min = np.sum(all_B_min !=0, axis=0) / num_runs
    fr_sel_min = pd.DataFrame(fr_sel_min,columns=['Frequency'])
    fr_sel_min.index = xnames.values.flatten()
    fr_sel_min = fr_sel_min.sort_values(by='Frequency', ascending=False)
    return fr_sel_min


def plot_top_features(fr_sel_min,numb_of_features=None): 
    """
    Sina Bonakdar, March 2025
    Plot the X top features (X is user set)
    """
    if numb_of_features==None:
        numb_of_features=25
    # Select the top features
    top_features = fr_sel_min.head(numb_of_features)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.barh(top_features.index, top_features['Frequency'], color='skyblue')
    plt.xlabel('Frequency of Selection')
    plt.ylabel('Features')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title('Frequency for Top {} Features in Resampling Models'.format(str(numb_of_features)))
    # Set tick parameters to place ticks inside the plot
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.show()


def PLSDA_model(X, Y, num_splits=None, num_components=None): 
    """
    Sina Bonakdar, March 2025
    perform n-fold CV and Return the average CV error and train error
    Y has to be one hot encoded
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import accuracy_score
    import numpy as np
    import pandas as pd

    if num_splits==None:
        num_splits=5
    if num_components==None:
        num_components=2
    if X.shape[1]==1:
        num_components=1
        
    # Initialize PLSRegression with an appropriate number of components
    pls = PLSRegression(n_components=num_components)
    # Cross-validation setup
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    # Get predictions for cross-validation
    y_cv_pred = cross_val_predict(pls, X, Y, cv=kf)
    # Calculate class predictions from continuous outputs
    y_cv_class = np.argmax(y_cv_pred, axis=1)
    y_true_class = np.argmax(Y.values, axis=1)
    # Calculate the cross-validation error (test error)
    cv_accuracy = accuracy_score(y_true_class, y_cv_class)
    cv_error = 1 - cv_accuracy
    # Now fit on the full dataset and predict to determine train error
    pls.fit(X, Y)
    y_train_pred = pls.predict(X)

    # Convert these to class predictions
    y_train_class = np.argmax(y_train_pred, axis=1)
    train_accuracy = accuracy_score(y_true_class, y_train_class)
    train_error = 1 - train_accuracy

    return train_error, cv_error



def find_best_num_features(xblock, yblock, fr_sel_min, n_splits=5, fraction_of_features=None): 
    """
    Sina Bonakdar, April 2025
    Find the best number of features with the lowest CV error.
    This function returns the best number of features along with train errors and test errors for all iterations.
    This is written in a way that doesn't raise error if it couldn't calculate error. 
    Instead, use the last train/test error.
    """
    import numpy as np

    # Define the number of features you want to evaluate in PLS analysis
    if fraction_of_features==None:
        fraction_of_features = 0.1  # Use 10% of features by default
        
    num_features = int(np.floor(fraction_of_features * np.shape(xblock)[1]))
    best_features = fr_sel_min.head(num_features)
    tr_err_lst = np.full((num_features, 1), np.nan)
    te_err_lst = np.full((num_features, 1), np.nan)

    # Stepwise feature adding for PLSDA
    for feat_id in range(0, num_features):
        sel_names = fr_sel_min.index[0:feat_id+1].values
        X = xblock[sel_names]
        Y = yblock
        
        try:
            tr_err, te_err = PLSDA_model(X, Y, num_splits=n_splits, num_components=2)
        except Exception as e:
            print(f'Exception occurred at feature {feat_id + 1}: {e}')
            # Use previous error values, or 0.5 for the first iteration if an error occurs
            if feat_id == 0:
                tr_err = 0.5
                te_err = 0.5
            else:
                tr_err = tr_err_lst[feat_id - 1]
                te_err = te_err_lst[feat_id - 1]

        # Record errors in lists
        tr_err_lst[feat_id] = tr_err
        te_err_lst[feat_id] = te_err

    # Select the best model based on the lowest CV error
    min_idx = int(np.argmin(te_err_lst, axis=0))
    print('Best number of features = ', min_idx + 1)
    
    return min_idx, tr_err_lst, te_err_lst



def plot_CV_Cal_err(best_sel, tr_err_lst, te_err_lst): 
    """
    Sina Bonakdar, March 2025
    plot the train and test errors vs. number of features selected
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    num_features = len(tr_err_lst)
    num_trys = np.arange(1, num_features+1)  # Replace with range (1, num_trys+1) if num_trys is a single int

    # Create the plot
    plt.figure(figsize=(8, 6))
    # Plot class_train (CalErr) and class_test (CVErr)
    p1, = plt.plot(num_trys, tr_err_lst, '-+', label='CalErr')
    p2, = plt.plot(num_trys, te_err_lst, '-+', label='CVErr')

    # Highlight the best selection point
    plt.plot(best_sel+1, tr_err_lst[best_sel], 's', markeredgecolor='k', markersize=10, markerfacecolor='none',
             label='Selected')
    plt.plot(best_sel+1, te_err_lst[best_sel], 's', markeredgecolor='k', markersize=10, markerfacecolor='none')
    # Add legend
    plt.legend()
    # Add labels and title
    plt.xlabel('Number of Features')
    plt.ylabel('Error')
    plt.title('Train Error vs Cross-Validated Error')
    # Customize font size of the axis
    plt.gca().tick_params(labelsize=14)
    # Show plot
    plt.show()


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



def PLSDA_plots(best_sel, fr_sel_min, xblock, yblock, tr_err_lst, te_err_lst, num_components=None, orthogonalize=False):
    """
    Sina Bonakdar, April 2025
    Fit the PLSDA model on the selected features using whole data and plot the scores and loadings plot
    best_sel,tr_err_lst,te_err_lst: first,second,third output of function find_best_num_features
    fr_sel_min: output of function freq_of_selection
    xblock, yblock from read_files function 
    if orthogonalize==True: orthogonalize the model.
    """
    sel_names = fr_sel_min.index[0:best_sel+1].values
    X = xblock[sel_names]
    Y = yblock
    feature_names = X.columns

    #plot scores and loadings using the whole dataset
    from sklearn.cross_decomposition import PLSRegression
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pyopls import OPLS
    import numpy as np
    # Initialize PLSRegression with an appropriate number of components
    if num_components==None:
        num_components=2

    #Orthogonolized
    if orthogonalize==True:
        Y_single_column = np.argmax(Y, axis=1)
        opls = OPLS(n_components=2)
        X = opls.fit_transform(X,Y_single_column)
        X = pd.DataFrame(X)
        X.columns= feature_names
    
    pls = PLSRegression(n_components=num_components)
    # Fit the model
    pls.fit(X, Y)
    # Extract the scores for the first two latent variables
    scores = pls.transform(X)
    # Extract loadings
    loadings = pls.x_loadings_
    # Create a DataFrame of loadings for easier handling
    #feature_names = X.columns
    loading_df = pd.DataFrame(loadings.T, columns=feature_names, index=['PLS1', 'PLS2'])
    # Select features for PLS1 and PLS2
    key_features_pls1 = loadings.T[0].argsort()[::-1] 
    key_features_pls2 = loadings.T[1].argsort()[::+1] 

    #Calculate the percent of variance captured
    perc_var_LV1, perc_var_LV2 = calc_perc_variance(X,pls)

    # Create color mapping
    color_mapping = {'BV': 'r', 'no BV': 'b'}

    # Set up a 2x2 plotting area
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # Plot (1, 1) - Scores plot with updated legend
    #sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=Y['Class2'], ax=axes[0, 0], palette=['b', 'r'])
    sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=Y['Class2'].map({1.0: 'BV', 0.0: 'no BV'}), ax=axes[0, 0], palette=color_mapping)
    axes[0, 0].legend(title='', edgecolor = 'black')
    axes[0, 0].set_title('PLS-DA Scores Plot')
    axes[0, 0].set_xlabel('Scores on LV1 ({}%)'.format(perc_var_LV1.round(2)))
    axes[0, 0].set_ylabel('Scores on LV2 ({}%)'.format(perc_var_LV2.round(2)))
    # Update the legend to use custom labels
    #handles, _ = axes[0, 0].get_legend_handles_labels()
    #axes[0, 0].legend(handles=handles, labels=['no BV', 'BV'], title='', edgecolor = 'black')
    # Plot dashed lines at x=0 and y=0
    axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    # Set ticks to point inside the plot
    axes[0, 0].tick_params(axis='both', direction='in')

    # Plot (1, 2) - Loadings plot for PLS2
    colors_pls2 = ['b' if val < 0 else 'r' for val in loadings.T[1, key_features_pls2]]
    sns.barplot(x=feature_names[key_features_pls2], y=loadings.T[1, key_features_pls2], ax=axes[0, 1], palette=colors_pls2)
    #axes[0, 1].set_title('Loadings on LV2')
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_ylabel('Loadings on LV2 ({}%)'.format(perc_var_LV2.round(2)))
    axes[0, 1].set_xticklabels(feature_names[key_features_pls2], rotation=45, ha='right')
    # Move y-label to the right side
    axes[0, 1].yaxis.set_label_position('right')
    axes[0, 1].yaxis.tick_right()
    # Set ticks to point inside the plot
    axes[0, 1].tick_params(axis='both', direction='in')

    # Plot (2, 1) - Loadings plot for PLS1
    colors_pls1 = ['b' if val < 0 else 'r' for val in loadings.T[0, key_features_pls1]]
    sns.barplot(x=loadings.T[0, key_features_pls1], y=feature_names[key_features_pls1], ax=axes[1, 0], palette=colors_pls1)
    #axes[1, 0].set_title('Loadings for PLS1')
    axes[1, 0].set_xlabel('Loadings on LV1 ({}%)'.format(perc_var_LV1.round(2)))
    axes[1, 0].set_ylabel('')
    # Set ticks to point inside the plot
    axes[1, 0].tick_params(axis='both', direction='in')

    # Hide the empty subplot (2, 2)
    axes[1, 1].axis('off')

    # Add a big title for the entire figure
    plt.suptitle('Selected PLSDA Model with {} Features'.format(best_sel+1), fontsize=24, fontweight='bold')

    # Adjust layout to handle overlapping plots
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)
    plt.show()

    print('Calibration Error: ',tr_err_lst[best_sel])
    print('CV Error: ',te_err_lst[best_sel])



def random_best_comparison(xblock, yblock, te_err_lst, best_sel, n_splits=5, num_rand_sets=None):
    """
    Sina Bonakdar, April 2025
    Compare the model performance with 500 sets including an equal size of random features with selected features.
    This helps to understand if the selected features using elastic net are meaningful or not.
    Perform a 2-sample t-test between CV accuracy of each random set and selected set in PLSDA model.
    This model also updated in a way that if tr/test errors couldn't be measured, just pass from it. 
    """
    import numpy as np
    from sklearn.utils import resample
    from scipy.stats import ttest_ind

    if num_rand_sets == None:
        num_rand_sets = 500  # Number of random sets to be generated 
    
    selected_feat_size = best_sel + 1
    CV_err_lst = np.full((num_rand_sets, 1), np.nan)
    
    for i in range(num_rand_sets):
        Y = yblock
        X = resample(xblock.T, n_samples=selected_feat_size, replace=False).T
        
        try:
            train_error, cv_error = PLSDA_model(X, Y, num_splits=n_splits, num_components=2)
            CV_err_lst[i] = cv_error
        except Exception as e:
            print(f'Exception occurred on iteration {i}: {e}')
            continue  # Skip to the next iteration if an error occurs

    # Compute accuracy metrics
    rand_CV_accuracy = 1 - CV_err_lst
    CV_accuracy_sel = 1 - te_err_lst[best_sel]
    rand_sig_pval = sum(rand_CV_accuracy > CV_accuracy_sel) / len(rand_CV_accuracy)
    print('Random feature selections beat selected features {}% of the times'.format(float(rand_sig_pval * 100)))

    # Perform 2-sample t-test
    rand_CV_accuracy = rand_CV_accuracy.flatten()
    rand_CV_accuracy = rand_CV_accuracy[~np.isnan(rand_CV_accuracy)]
    t_stat, p_value = ttest_ind(rand_CV_accuracy, float(CV_accuracy_sel))
    # Print results
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")
    
    return rand_CV_accuracy, CV_accuracy_sel



def plot_cv_accuracy(rand_CV_accuracy, CV_accuracy_sel):
    """
    Sina Bonakdar, March 2025
    plot random accuracies vs. best_model accuracy
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # Create a new figure with a white background
    plt.figure(figsize=(8, 6), facecolor='white')
    # Scatter plot for random model accuracies with jitter
    temp = rand_CV_accuracy
    x_jitter = 1 + (np.random.rand(temp.size) - 0.5) / 10
    plt.plot(x_jitter, rand_CV_accuracy, 'o', 
             markerfacecolor=[0.3010, 0.7450, 0.9330],
             markeredgecolor=[0, 0.4470, 0.7410], 
             linestyle='None')
    # Line plot for the accuracy of the selected model
    plt.plot([0.9, 1.1], [CV_accuracy_sel, CV_accuracy_sel], '-m', linewidth=2)
    
    # Set plot limits and labels
    plt.xlim([0.7, 1.3])
    plt.ylim([0, 1.2])
    plt.xticks([1], ['Comparison'])
    plt.title(f'CV performance of selected model to {rand_CV_accuracy.shape[0]} random models of equal size')
    plt.ylabel('Accuracy')
    # Set ticks to point inside the plot
    plt.tick_params(axis='both', direction='in', colors='black')
    # Turn off grid
    plt.grid(False)
    # Adding legend
    plt.legend(['Random', 'Selected Model'], loc='upper right')
    # Display the plot
    plt.show()