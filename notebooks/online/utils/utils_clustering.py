# here we store some functions, which are used in clustering notebook

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import skfuzzy as fuzz
from itertools import groupby
from collections import Counter
from dictionaries_rename import *
from PIL import Image


#-------------------------PLOTTING-------------------------#
def save_fig(title):
    plt.savefig("../../../figures/clustering_plots/"+title +
                ".png", bbox_inches='tight', dpi=600)

def plot_metrics(metric, wave, p, title): 
    plt.plot(['Kmeans', 'Ward', 'Birch', 'Gaussian Mix.', 'Agglomerative Cl.'],
             metric, marker='o', label="wave " + wave)
    plt.grid(True)
    plt.gcf().set_size_inches(15, 8)
    plt.xticks(rotation=15)
    plt.legend()
    plt.gca().set_title(title)
    plt.subplot(1, 3, p)
    
def plot_optimal_k(metrics_k, K_list, title, x_label, y_label):
    """plot and save metrics chart for Elbow"""

    # styling
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    plt.rcParams["figure.figsize"] = 5, 7

    # Plot multiple lines
    num = 0
    for column in metrics_k:
        num += 1
        plt.plot(metrics_k.index, metrics_k[column], marker='', color=palette(
            num), linewidth=5, alpha=0.5, label='Wave: '+column)

    # add decorations
    plt.legend(ncol=2, fontsize=11, loc='upper right')
    plt.title(title,
              loc='left', fontsize=18, fontweight=0, color='royalblue')
    plt.xlabel(x_label, fontsize=15)
    plt.xticks(K_list)
    plt.ylabel(y_label, fontsize=15)

    save_fig(title)
    plt.show()
    
def plot_left_right(left_right, df_plot, centroids_transformed, wave):
        
    # prepare continious scale for legend
    df_lr = pd.concat([df_plot, left_right[wave]], axis=1) 
    norm = plt.Normalize(df_lr.iloc[:, 4].min(), df_lr.iloc[:, 4].max())
    # coolwarm_r: reversed colormap, 0 is left, 10 is right
    sm = plt.cm.ScalarMappable(cmap="coolwarm_r", norm=norm)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = sns.scatterplot(data=df_lr,
                              x=df_lr['1st component'], y=df_lr['2nd component'],
                              hue=df_lr.iloc[:, 4], palette='coolwarm_r', style="Label", s=30)
    centr = sns.scatterplot(centroids_transformed[:, 0], centroids_transformed[:, 1],
                            marker='+', color='black', s=300)
    n_clusters = df_lr['Label'].nunique()
    title = f'{left_right[wave].columns[0]}, wave {wave}, {n_clusters} clusters'
    # add annotations to the continious left-right legend
    ax.annotate('Left', xy=(1.01, 0), xycoords='axes fraction', fontsize=10,
                horizontalalignment='left', verticalalignment='bottom')
    ax.annotate('Right', xy=(1.01, 1), xycoords='axes fraction', fontsize=10,
                horizontalalignment='left', verticalalignment='top')
    
    plt.title(title, size=15)
    scatter.figure.colorbar(sm)
    # showing only labels legend (without colors, which are shown on separate legend)
    h, l = scatter.get_legend_handles_labels()
    plt.legend(h[7:10], l[7:10], loc=2, borderaxespad=0.)
    # save png
    save_fig(title)
    # plots are not printed due to space constraints, use widgets (we leave just wave 1 for demonstration)
    if wave != '1':
        plt.close()

def plot_party_choice(party_choice, df_plot, centroids_transformed, wave):
    df_parties = pd.concat([df_plot, party_choice[wave]], axis=1) 
    color_discrete_map = {'SPOE': '#f54242', 
                              "List Sebastian Kurz: OEVP": '#42ecf5',
                              'FPOE': '#423bd9',
                              'The Greens': '#3bd93d',
                              'NEOS': '#ff4acc',
                              'Team Stronach': '#696969'}           
    fig, ax = plt.subplots(figsize=(12, 8))
    df_parties = df_parties[df_parties['PARTY CHOICE: PROSPECTIVE'].isin(['SPOE', 
                                                                       "List Sebastian Kurz: OEVP",
                                                                       'FPOE',
                                                                       'The Greens',
                                                                       'NEOS',
                                                                       'Team Stronach'])]
    scatter = sns.scatterplot(data=df_parties,
                                x=df_parties['1st component'], y=df_parties['2nd component'],
                                hue=df_parties['PARTY CHOICE: PROSPECTIVE'], palette=color_discrete_map, style="Label", s=30)
    centr = sns.scatterplot(centroids_transformed[:, 0], centroids_transformed[:, 1],
                            marker='+', color='black', s=300)
    n_clusters = df_parties['Label'].nunique()
    title = f'Party choice (prospective), wave {wave}, {n_clusters} clusters'
    plt.title(title, size=15)
    # save png
    save_fig(title)
    # plots are not printed due to space constraints, use widgets (we leave just wave 1 for demonstration)
    if wave != '1':
        plt.close()
        
def elements_info(mean_cluster, n_samples):
    print('Rate of those who never flip-flop: ', round(((mean_cluster ==
          1).sum()+(mean_cluster == 0).sum())/mean_cluster.shape[0], 2))
    print('Samples on the plot: ', round(mean_cluster.shape[0]/n_samples, 2))

def plot_mean_label(df, title, n_samples):
    """finds mean among labels for waves, when one has participated in survey, creates bar plot"""

    mean_cluster = df.filter(like='Label w', axis=1).mean(axis=1)
    elements_info(mean_cluster, n_samples)
    sns.displot(data=mean_cluster)
    plt.title(title, size=15)
    save_fig(title)
    
def real_or_nan(wave):
    """as we were replacing NaN values with mode mostly, we cannot be sure if we see real value on plots or it was NaN initially
    (for instance, if we have many values equal to 5 in self left-right placement, does that mean, that many people put themselves
    to the center? Or maybe that big number of center values was obtained artificially?),
    therefore we plot distribution of value counts for feature plotted"""
    df_init = pd.read_csv('../../../data/raw/10017_da_en_v2_0.tab', sep='\t')
    questions = {'1': 'w1_q9', '2': 'w2_q35x2', '3': 'w3_q40x2', '4': 'w4_q40x2', '5': 'w5_q6', '6': 'w6f_q13'}
    dict_ord = get_ordinal_names()
    plt.title(f'{dict_ord.get(questions.get(wave))} distribution')
    df_init[questions.get(wave)].value_counts(dropna=False).plot(kind='barh')

def plot_clusters(wave, clusters, left_right):
    """Shows plots with samples distribution among clusters with left-right and party-choice coloring"""
    real_or_nan(wave)
    left_right_image = Image.open(
        f"../../../figures/clustering_plots/{left_right[wave].columns[0]}, wave {wave}, {clusters} clusters.png"
    )
    left_right_image = left_right_image.resize((800, 600))
    display(left_right_image)
    if wave in {"1", "2", "3", "4"}:
        party_choice_image = Image.open(
            f"../../../figures/clustering_plots/Party choice (prospective), wave {wave}, {clusters} clusters.png"
        )
        party_choice_image = party_choice_image.resize((800, 600))
        display(party_choice_image)
        
def plot_stability(include, exclude_wave_6, df_clustered, df_close):
    """bar plot with mean cluster labels"""
    n_samples = df_clustered.shape[0]
    if exclude_wave_6 == True:
        df_clustered_ = df_clustered.drop(["Label w6"], axis=1)
        df_close_ = df_close.drop(["Label w6"], axis=1)
        if include == "always closely located samples":
            plot_mean_label(
                df_close_,
                "Label among clusters (close to centroids during waves 1-5)",
                n_samples,
            )
        else:
            df_clustered["Times"] = df_clustered_.filter(like="Close", axis=1).sum(
                axis=1
            )
            df_clustered_ = df_clustered_.loc[df_clustered["Times"] >= 1]
            plot_mean_label(
                df_clustered_,
                "Label among clusters (close to centroids at least once during waves 1-5)",
                n_samples,
            )
    else:
        if include == "always closely located samples":
            plot_mean_label(
                df_close,
                "Label among clusters (close to centroids during all waves of participation)",
                n_samples,
            )
        else:
            plot_mean_label(
                df_clustered,
                "Label among clusters (close to centroids at least once during all waves of participation)",
                n_samples,
            )
    
def plot_deviations(opinion_deviations):
    font_color = "#525252"
    csfont = {"fontname": "Calibri"}  # title font
    hfont = {"fontname": "Calibri"}  # main font

    ax = opinion_deviations.T.plot.barh(
        align="center", stacked=True, figsize=(15, 20)
    ) 
    plt.tight_layout()

    title = plt.title(
        "Deviations in opinions", pad=60, fontsize=18, color=font_color, **csfont
    )

    # Adjust the subplot so that the title would fit
    plt.subplots_adjust(top=0.8, left=0.26)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)
    plt.xticks(color=font_color, **hfont)
    plt.yticks(color=font_color, **hfont)

    legend = plt.legend(
        loc="center",
        frameon=False,
        bbox_to_anchor=(0.0, 0.97, 1.0, 0.102),
        mode="expand",
        ncol=6,
        borderaxespad=-0.46,
        prop={"size": 15, "family": "Calibri"},
    )

    for text in legend.get_texts():
        plt.setp(text, color=font_color)  # legend font color

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(
            x + width / 2,
            y + height / 2,
            "{:.0f}".format(width),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=13,
            **hfont
        )
    title = "Deviations in opinions"
    save_fig(title)
    
def individual_check(df_clustered, waves):
    """Heatmap showing individual labels assignment"""
    df_clustered_close = df_clustered.copy()
    # filtering 'confidently' clusteres samples (if they participated in all waves and always were close to centroids)
    for wave in waves:
        df_clustered_close = df_clustered_close.loc[
            df_clustered_close[f"Close to centroid w{wave}"] == 1
        ]

    f, ax = plt.subplots(figsize=(25, 5))
    # Define colors
    colors = ["midnightblue", "mediumspringgreen"]
    cmap = LinearSegmentedColormap.from_list("Custom", colors, len(colors))
    df_labels = df_clustered_close.filter(like="Label", axis=1)
    df_labels["mean_label"] = df_labels.mean(axis=1)
    df_labels.sort_values(by="mean_label", ascending=False, inplace=True)
    df_labels.drop(columns=["mean_label"], inplace=True)
    ax = sns.heatmap(df_labels.T, cmap=cmap)
    # Set the colorbar labels
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(["0", "1"])
    title = "Consistency of cluster assignments"
    plt.title(title, fontsize=15)
    save_fig(title)

#-------------------------DATA MANIPULATION-------------------------#
def prettify_feature_names(df):
    """Get rid of OPINION tag and feature codes for printing due to space constraints"""
    df.columns = df.columns.str.split('-w').str[0]
    df.columns = df.columns.str.split('OPINION: ').str[1]
    return df

def tag_immigration(df):
    """adds 'im: ' to each question name, which relates to immigration (using markers for list below""" 
    markers = ['IMMIGR', 'MUSL', 'ASYL', 'NON-AUSTR', 'SOCIAL BENEFITS: EASTERN EUROPEANS', 'SOCIAL BENEFITS: WESTERN EUROPEANS', 'FREEDOM OF MOVEMENT IN EU']
    imm_questions = [question for question in df.columns if any(marker in question for marker in markers)]
    for question in imm_questions:
        df = df.rename(columns={question: 'im: ' + question})
    return df

def sort_by_absolute_val(df, column):
    """Sort df column by descending order in terms of the absolute value."""
    df = df.reindex(df[column]
                    .abs()
                    .sort_values(ascending=False)
                    .index)
    return df

def back_from_dummies(df):
    """reverting dummy features back to single column"""
    
    result_series = {}

    # Find dummy columns and build pairs (category, category_value)
    dummmy_tuples = [(col.split("__")[0],col) for col in df.columns if "__" in col]

    # Find non-dummy columns that do not have a _
    non_dummy_cols = [col for col in df.columns if "__" not in col]

    # For each category column group use idxmax to find the value.
    for dummy, cols in groupby(dummmy_tuples, lambda item: item[0]):

        #Select columns for each category
        dummy_df = df[[col[1] for col in cols]]

        # Find max value among columns
        max_columns = dummy_df.idxmax(axis=1)

        # Remove category_ prefix
        result_series[dummy] = max_columns.apply(lambda item: item.split("__")[1])

    # Copy non-dummy columns over.
    for col in non_dummy_cols:
        result_series[col] = df[col]

    return pd.DataFrame(result_series)

def store_results(df, explained_variance, eigenvectors, strongest_differences, wave):
    """store the results in the nested dict"""
    clustering_info = {}
    clustering_info[f"Table 1. Explained variance, wave {wave}"] = explained_variance
    clustering_info[
        f"Table 2. Important features (PCA), 1 component, wave {wave}"
    ] = sort_by_absolute_val(eigenvectors, "Eigenvector 1")["Eigenvector 1"].to_frame()
    clustering_info[
        f"Table 3. Important features (PCA), 2 component, wave {wave}"
    ] = sort_by_absolute_val(eigenvectors, "Eigenvector 2")["Eigenvector 2"].to_frame()
    clustering_info[
        f"Table 4. The most important factors regarding differences between clusters (1 minus 0), wave {wave}"
    ] = strongest_differences
    return clustering_info

def show_features(wave, clusters, results_2, results_4):
    """Show explained variance, features, forming eigenvectors and those having the largest differences between clusters"""
    if clusters == 2:
        # looping over nested dicts
        wave_dict = results_2[wave]
        for key, values in wave_dict.items():
            for key_, values_ in values.items():
                display(values_[:5].style.set_caption(key_))

    else:
        # looping over nested dicts
        wave_dict = results_4[wave]
        for key, values in wave_dict.items():
            for key_, values_ in values.items():
                if (
                    key_
                    == f"Table 4. The most important factors regarding differences between clusters (1 minus 0), wave {wave}"
                ):
                    for key__, values__ in values_.items():
                        display(
                            values__.style.set_properties(
                                **{"background-color": "#000066", "color": "white"},
                                subset=[f"mean_{key__}"],
                            )
                        )
                else:
                    display(values_[:5].style.set_caption(key_))

def delete_unique(cols_names):
    """Returns duplicated elements from list as we want only repeated questions"""
    duplicated_cols = [
        question for question in cols_names if cols_names.count(question) > 1
    ]
    duplicated_cols = list(dict.fromkeys(duplicated_cols))
    return duplicated_cols
                    

#-------------------------CLUSTERS MANIPULATION-------------------------#
def compare_4_clusters(df, wave):
    """Computes mean opinions among classes,
    as well as standart deviation, prints 10 questions with the largest distance
    from mean among classes in descending order.

    :input: 
    - df with opinions for each sample and every question merged with cluster labels. 
    - wave.

    :return:
    - df with the questions (as indexes), which contribute to the largest difference 
            between mean value of cluster $j$ and mean of means of other clusters. 
            Questions are sorted in descending order. Mean value of cluster $j$, then its difference 
            from other 3 clusters and std are provided in columns.

    """

    # compute mean and std for every cluster
    opinion_stats = df.groupby(['Label']).mean().T
    opinion_std = df.groupby(['Label']).std().T
    opinion_std.columns = [str(i) + '_std' for i in opinion_std.columns]
    labels_number = len(Counter(df.Label))

    # find the difference of mean inside cluster from the mean of the three out-of-class means
    for label in range(labels_number):
        opinion_stats['diff_mean' +
                      str(label)] = opinion_stats[label] - opinion_stats.drop(label, axis=1).mean(axis=1)

    opinion_stats = pd.concat([opinion_stats, opinion_std], axis=1)
    # sort the most important factors by abs value of difference (with sign remain unchanged eventually)
    opinion_stats_sorted_all = {}
    for label in range(labels_number):

        # exclude label column to compute difference with other means
        for other_class_label in range(labels_number):
            opinion_stats[f'diff_means{other_class_label}'] = opinion_stats[other_class_label] - \
                opinion_stats[label]

        opinion_stats_sorted = sort_by_absolute_val(
            opinion_stats, f'diff_mean{label}')

        # renaming 0,1,2,3 to mean_0 etc
        opinion_stats_sorted.columns.values[[0, 1, 2, 3]] = [
            'mean_0', 'mean_1', 'mean_2', 'mean_3']

        # drop cols with differences of mean_i and mean_j (with 0 entries)
        opinion_stats_sorted = opinion_stats_sorted.loc[:, (opinion_stats_sorted != 0).any(
            axis=0)]

        mean_inside_cluster = opinion_stats_sorted.filter(
            like=f'mean_{label}', axis=1)
        differences_out = opinion_stats_sorted.filter(
            like='diff_means', axis=1)
        std_all = opinion_stats_sorted.filter(like='std', axis=1)
        info_to_display = pd.concat(
            [mean_inside_cluster, differences_out, std_all], axis=1)
        opinion_stats_sorted_all[label] = info_to_display[:5]
    return opinion_stats_sorted_all

def waves_to_switch(df, wave):
    """in order to compare cluster assignment stability we would
    like to be sure that labels assigned have the same magnitude
    (otherwise the same cluster might have label 1 for one wave and
    label 0 for another one"""

    if wave == str(2):
        if df.loc['im: FEELING LIKE A STRANGER DUE TO THE MANY MUSLIMS ', 'Difference'] > 0:
            print(f'Switch label signs for wave: {wave}')
            return wave
    elif wave == str(6):
        if df.loc['POLITICIANS DO NOT CARE ABOUT WHAT PEOPLE LIKE ME THINK ', 'Difference'] > 0:
            print(f'Switch label signs for wave: {wave}')
            return wave
    else:
        if df.loc['im: CRIME RATES INCREASE IN AUSTRIA BECAUSE OF IMMIGRANTS ', 'Difference'] > 0:
            print(f'Switch label signs for wave: {wave}')
            return wave
        


#-------------------------SOFT CLUSTERING-------------------------#
# depreciated since we could not obtain more stable results in comparison with hard K-means

# Scikit-Fuzzy is a collection of fuzzy logic algorithms intended for use in
# the SciPy Stack, written in the Python computing language.
# https://pythonhosted.org/scikit-fuzzy/overview.html
# !conda install -c conda-forge scikit-fuzzy

# Soft K-means works similarly to standard K-means, returning not labels, but probabilities for sample to be assigned to each cluster. By tuning the threshold we can select people, who are confidently assigned to some cluster (while in K-means we filtered samples depending on their closeness to centroids). The graph below shows consistency by "mean label" among the waves, similarly to plot above.
"""
def partition_matrices_(df_opinion, waves):
    partition_matrices = pd.DataFrame([])
    df_confidence_wave = {}
    for wave in waves:
        # fitting soft K-means
        # https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.cluster.html#cmeans
        cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
            df_opinion[wave].T, 2, 1.5, error=0.005, maxiter=1000, seed=42)
        # predicting labels
        # https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.cluster.html#cmeans-predict
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            df_opinion[wave].T, cntr, 1.5, error=0.005, maxiter=1000, seed=42)
        df_confidence_wave[wave] = pd.DataFrame(u.T, columns=[
                                             f'Confidence cl. 1, wave. {wave}', f'Confidence cl. 2, wave. {wave}'], index=df_opinion[wave].index)
        partition_matrices = pd.concat(
            [partition_matrices, df_confidence_wave[wave]], axis=1)
        
    return partition_matrices


partition_matrices = partition_matrices_(df_opinion, waves)
@interact
def plot_fuzzy(threshold=[0.5, 0.65, 0.8]):

    # filtering confidently (as p>=threshold for cluster, else NaN) clustered samples,
    # replacing probabilities by labels
    confident_labels = partition_matrices.filter(like='Confidence cl. 1', axis=1)
    def remap(x):
        if x <= (1-threshold):
            return 0
        elif x > (1-threshold) and x < threshold:
            return np.nan
        else:
            return 1

    confident_labels = confident_labels.applymap(remap)
    confident_labels.columns = [f'Label w{w}' for w in waves]
    # drop rows with NaN labels (which are not confident)
    confident_labels = confident_labels.dropna() #subset=['Label']
    list_to_switch = []
    # add cluster label column
    for w in waves:
        soft_label = confident_labels.filter(like=w, axis=1)
        soft_label.columns = ['Label']
        df_soft_labels = pd.concat([df_opinion[w], soft_label], axis=1)
        #excl = df_soft_labels['Label'].isna().sum()/df_clustered.count(axis=0)[int(wave)*2-1]
        #print(f'Rate of not confident samples, wave {w}: ', excl.round(2), '%', sep='')
        fuzzy_factors, waves_to_switch = cluster_differences(df_soft_labels, w)
        list_to_switch.append(waves_to_switch)
    confident_labels = switch_labels(list_to_switch, confident_labels)  
    mean_cluster = confident_labels.filter(
        like='Label w', axis=1).mean(axis=1)
    n_samples = df_clustered.shape[0]
    print('Rate of those who never flip-flop: ', round(((mean_cluster == 1).sum()+(mean_cluster == 0).sum())/mean_cluster.shape[0], 2))
    print('Samples on the plot: ', round(mean_cluster.shape[0]/n_samples, 2))
    sns.displot(data=mean_cluster)
    title = 'Soft Kmeans (all samples)'
    plt.title(title, size=15)
    save_fig(title)
"""