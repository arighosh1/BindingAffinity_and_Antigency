import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import streamlit as st
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from PIL import Image
from quantiprot.metrics.aaindex import get_aa2charge, get_aa2hydropathy
from PIL import Image
img = Image.open('icon.jpeg')
st.set_page_config(page_title='RATIONAL VACCINE DESIGN FOR VIRUS USING MACHINE LEARNING APPROACHES', page_icon=img,
                   layout='wide', initial_sidebar_state='auto')
# import os
# #
# basedir = os.path.dirname(os.path.abspath(__file__))
# basedir += "/Training set 3481 PDB_IDs.csv"
# st.markdown(
#
#     """
#     <style>
#     .reportview-container {
#         background-color: #003366;
#
#     }
#    .sidebar .sidebar-content {{
#                 background-color: #003366;
#             }}
#     </style>
#     """,
#     unsafe_allow_html=True
#  )
st.markdown(
    '''
        <style>
            @media (max-width: 991.98px)
            {
                .sidebar .sidebar-content 
                {
                    background-color: #003366;
                }
            }
        </style>
    ''',
    unsafe_allow_html=True
)
st.sidebar.image("https://github.com/arighosh1/BindingAffinity_and_Antigency/blob/main/icon.jpeg?raw=true")
value=st.sidebar.slider("Slide to 1 for Binding Affinity And 2 for Antigenicity",0,1,0)
if value == 0:
    # !/usr/bin/env python
    # coding: utf-8

    # In[1]:

    import os
    import warnings

    warnings.filterwarnings("ignore")

    # In[2]:

    # import libraries
    import streamlit as st
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn import model_selection
    from scipy.stats import spearmanr
    from scipy.stats import pearsonr
    import math
    from math import sqrt
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn import model_selection
    from scipy.stats import spearmanr
    from scipy.stats import pearsonr
    import math
    from math import sqrt
    import pandas as pd
    import numpy as np
    import requests

    # In[3]:

    file = st.file_uploader("Upload CSV File : ", accept_multiple_files=False)

    if file != None:
        # Read the data
        df_TR = pd.read_csv("https://raw.githubusercontent.com/arighosh1/BindingAffinity_and_Antigency/main/Training%20set%203481%20PDB_IDs.csv")
        df_TS = pd.read_csv(file)

        # st.write("Training Set Data : ")
        # st.dataframe(df_TR)
        # st.write("Testing Set Data : ")
        # st.dataframe(df_TS)
        # In[4]:



        # In[8]:

        # Traning Sets
        y_df_TR = df_TR['pKd']
        X_df_TR = df_TR.drop(['PDB_ID', 'Resolution', 'pKd'], axis=1)

        # In[9]:

        # X_df_TR.shape, y_df_TR.shape

        # In[10]:

        X_train, X_valid, Y_train, Y_valid = train_test_split(X_df_TR, y_df_TR, test_size=0.2,random_state=123456)

        # In[11]:

        # Test Sets
        y_df_TS = df_TS['pKd']
        X_df_TS = df_TS.drop(['PDB_ID', 'Resolution', 'pKd'], axis=1)

        # In[12]:

        # # Optimized parameters
        # ## max_features = 'auto'
        # ## n_estimators=100
        # ## random_state = 1234

        # In[13]:

        models_RF_train = {"RF": RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                                                       max_features='auto', max_leaf_nodes=None,
                                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                                       min_samples_leaf=1, min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0, n_estimators=100,
                                                       n_jobs=None, oob_score=False, random_state=1234,
                                                       verbose=0, warm_start=False)}

        # In[14]:

        # Calculate the Training and Validation (Refined set) statistics
        scores = {}
        for m in models_RF_train:
            models_RF_train[m].fit(X_train, Y_train)
            scores[m + "_train_r2"] = models_RF_train[m].score(X_train, Y_train)
            Y_pred_valid_rf = models_RF_train[m].predict(X_valid)
            Y_pred_train_rf = models_RF_train[m].predict(X_train)
            scores[m + "_rmse_train"] = sqrt(mean_squared_error(Y_train, Y_pred_train_rf))
            scores[m + "_mae_train"] = mean_absolute_error(Y_train, Y_pred_train_rf)
            scores[m + "_pcc_train"] = pearsonr(Y_train, Y_pred_train_rf)
            scores[m + "_valid_r2"] = r2_score(Y_valid, Y_pred_valid_rf)
            scores[m + "_rmse_valid"] = sqrt(mean_squared_error(Y_valid, Y_pred_valid_rf))
            scores[m + "_mae_valid"] = mean_absolute_error(Y_valid, Y_pred_valid_rf)
            scores[m + "_pcc_valid"] = pearsonr(Y_valid, Y_pred_valid_rf)

        scores_RF_train = pd.Series(scores).T
        # scores_RF_train

        # In[15]:
        # Calculate statistics for test set (Core set) based on RF model
        scores = {}
        for m in models_RF_train:
            Y_pred_test_rf = models_RF_train[m].predict(X_df_TS)
            scores[m + "_test_r2"] = r2_score(y_df_TS, Y_pred_test_rf)
            scores[m + "_rmse_test"] = sqrt(mean_squared_error(y_df_TS, Y_pred_test_rf))
            scores[m + "_mae_test"] = mean_absolute_error(y_df_TS, Y_pred_test_rf)
            scores[m + "_pcc_test"] = pearsonr(y_df_TS, Y_pred_test_rf)

        scores_RF_test = pd.Series(scores).T
        # scores_RF_test

        # In[16]:

        # Save the test prediction result
        Pred_y = pd.DataFrame({'Y_pred_rf': Y_pred_test_rf})
        Exp_y = pd.DataFrame(y_df_TS)
        Prediction = pd.concat([Exp_y, Pred_y], axis=1)
        st.title("Result of RandomForestRegressor.")
        st.write(Prediction)


        # In[34]:

        YV_array = np.array(Y_valid)
        YT_array = np.array(Y_train)
        XV_array = np.array(X_valid)
        XT_array = np.array(X_train)

        # In[24]:

        from sklearn.neighbors import KNeighborsRegressor

        knn_model = KNeighborsRegressor(n_neighbors=3)

        # In[25]:

        knn_model.fit(XT_array, YT_array)

        # In[27]:

        from sklearn.metrics import mean_squared_error
        from math import sqrt

        train_preds = knn_model.predict(XV_array)

        # In[31]:

        # print("KNN predicted Vlue:", train_preds)

        st.title("Result of KNN : ")
        Pred_y = pd.DataFrame({'Y_pred_KNN': train_preds})
        Exp_y = pd.DataFrame(y_df_TS)
        Prediction = pd.concat([Exp_y, Pred_y], axis=1)
        st.write(Prediction)
        # In[33]:

        mse = mean_squared_error(YV_array, train_preds)
        rmse = sqrt(mse)
        # print("RMSE_train KNN:", rmse)

        st.write("RMSE_train KNN:", rmse)

        # In[ ]:

        # In[ ]:






elif value == 1:
    import streamlit as st
    # !/usr/bin/env python
    # coding: utf-8

    # # Hopp-Woods Hydrophilicity Prediction with Linear Variation Model

    # ## Description
    #
    # A numeric value is assigned to each amino acid, using the amino acid scale outlined in the paper:
    #
    # *Prediction of protein antigenic determinants from amino acid sequences*, T Hopp, K Woods, PNAS 1981
    #
    # User provides protein seq in single-letter amino acid code, specifies a window size (length of the peptide), and edge weight (default $\alpha=1$). For each amino acid in the window, the program computes a weight using the linear variation model. It then applies the weight to the original score at amino acid level. The final hydrophilicity score for the peptide is calculated by dividing the sum of the corrected amino acid scores by the sum of the weights. The program repeats the process along the sequence of the protein.

    # ## Mathematical principles
    #
    # Given:
    #
    # $S=\Bigg\{\phi(n)=\frac{\sum\limits_{i=n}^{n+\Delta-1} w_{i}X_{i}}{\sum\limits_{i=n}^{n+\Delta-1} w_{i}} \Bigg| 0\le n \le N-\Delta\Bigg\}$
    #
    # Rank items in set $S$ from high to low
    #
    # where:
    #
    # $\phi(n)$: Weighted ($w_{i}\neq1$) or non-weighted ($w_{i}=1$) hydrophilicity scores
    #
    # $N$: Number of amino acids in the protein
    #
    # $n$: residue index position on the protein (starting from 0)
    #
    # $\Delta$: size of the peptide "window"
    #
    # $X_{i}$: Hopp-Woods hydrophilicity value of amino acid $X$
    #          at index position $i$
    #
    # $w_{i}$: weight used at each position. Weights are calculated using linear variation model (see below)

    # ## Linear Variation Model for Calculation of Weights
    #
    # 1. When no weights are used:
    #
    #    $$w_{i}=1$$
    #
    # 2. When using weights from linear variation model, specify edge weight $\alpha (0<\alpha\le1$)
    #
    #    1) When the peptide window ($\Delta$) is an odd number:
    #
    #    $w_{i}=
    # \begin{cases}
    # \alpha+\frac{1-\alpha}{\lfloor 0.5\Delta \rfloor}q & 0\le q \le \lfloor 0.5\Delta \rfloor\\
    # 1-\frac{1-\alpha}{\lfloor 0.5\Delta \rfloor}(q-\lfloor 0.5\Delta \rfloor) & \lfloor 0.5\Delta \rfloor\ < q \le \Delta-1
    # \end{cases}
    # $
    #
    #   For example, if window=7 (7-mer peptide), edge $\alpha=0.1$, then the first and the last weights will be 0.1. The weight for each amino acid in the 7-mer is:
    #
    #   [0.1, 0.4, 0.7, 1.0, 0.7, 0.4, 0.1]
    #
    #   2) When the peptide window ($\Delta$) is an even number:
    #
    #      $w_{i}=
    # \begin{cases}
    # \alpha+\frac{1-\alpha}{0.5\Delta-1}q & 0\le q < 0.5\Delta\\
    # 1-\frac{1-\alpha}{0.5\Delta-1}(q-0.5\Delta) & 0.5\Delta \le q \le \Delta-1
    # \end{cases}
    # $
    #
    #   For example, if window=10, edge $\alpha=0.1$, then the first and the last weights will be 0.1.The weight for each amino acid in the 10-mer is:
    #
    #   [0.1, 0.33, 0.55, 0.78, 1.0, 1.0, 0.78, 0.55, 0.32, 0.1]

    # In[9]:

    import matplotlib.pyplot as plt

    #
    #
    # fig, ax=plt.subplots(1, 3, sharey=True, figsize=(12, 6))
    #
    # ax[0].plot([x for x in range (1, 8)],
    #            [1 for x in range (1, 8)], linewidth=7, alpha=0.4)
    # ax[0].set_xlabel("Amino Acid Position")
    # ax[0].set_ylabel("Weight")
    # ax[0].set_title("Widow size=7, alpha=1 (No Weight Applied)")
    #
    # ax[1].plot([x for x in range (1, 8)],
    #          [0.1, 0.4, 0.7, 1.0, 0.7, 0.4, 0.1], linewidth=7, alpha=0.4)
    # ax[1].set_xlabel("Amino Acid Position")
    # ax[1].set_title("Widow size=7, alpha=0.1")
    #
    # ax[2].plot([x for x in range (1, 11)] ,
    #          [0.1, 0.33, 0.55, 0.78, 1.0, 1.0, 0.78, 0.55, 0.32, 0.1], linewidth=7, alpha=0.4)
    # ax[2].set_xlabel("Amino Acid Position")
    # ax[2].set_title("Widow size=10, alpha=0.1")
    #
    # fig.suptitle("Weights Computed by Linear Variation Model")
    #
    # fig.savefig("Weights by Linear Variation Model", dpi=300)
    #
    #
    # # ## Code
    #
    # # In[1]:
    #

    import matplotlib.pyplot as plt

    # Amino Acid Scale Defined by Hopp-Woods's original paper
    hopp_scores = {
        "R": 3,
        "D": 3,
        "E": 3,
        "K": 3,
        "S": 0.3,
        "N": 0.2,
        "Q": 0.2,
        "G": 0,
        "P": 0,
        "T": -0.4,
        "A": -0.5,
        "H": -0.5,
        "C": -1,
        "M": -1.3,
        "V": -1.5,
        "I": -1.8,
        "L": -1.8,
        "Y": -2.3,
        "F": -2.5,
        "W": -3.4
    }


    def lm(pept_length, alpha):
        """
        Compute weights using linear variation model
        :param pept_length: int
                size of the window
        :param alpha: float between 0 (exclusive) and 1 (inclusive)
                edge weight

        :return: list
                a list of weights.
        """
        weight_lst = []
        if pept_length % 2 != 0:
            for idx in range(0, pept_length):
                if idx <= pept_length // 2:
                    weight = alpha + (1 - alpha) * idx / (pept_length // 2)
                    weight = round(weight, 2)
                    weight_lst.append(weight)
                else:
                    weight = 1 - (1 - alpha) * (idx - pept_length // 2) / (pept_length // 2)
                    weight = round(weight, 2)
                    weight_lst.append(weight)
        else:
            for idx in range(0, pept_length):
                if idx < pept_length / 2:
                    weight = alpha + (1 - alpha) * idx / (pept_length / 2 - 1)
                    weight = round(weight, 2)
                    weight_lst.append(weight)
                else:
                    weight = 1 - (1 - alpha) * (idx - pept_length / 2) / (pept_length / 2 - 1)
                    weight = round(weight, 2)
                    weight_lst.append(weight)

        return weight_lst


    def calc_hopp(seq, pep_length, alpha=1):
        """
        Calculate the hopp-woods score for each peptide using the linear variation model
        :param seq:str
                protein seq in one-letter code
        :param pep_length:int
                size of the window (length of the peptide)
        :param alpha: float
                edge weight, between 0 (exclusive) and 1 (inclusive)
        :return:tuple
                a tuple (averaged hydrophilicity score, peptide seq)
        """

        # Caculate un-corrected score
        aa_lst = list(seq)
        resi_hopp_lst = [hopp_scores[x] for x in aa_lst]

        # Caculate weights
        weight_lst = lm(pep_length, alpha)
        st.write("Weights used: ", end="")
        st.write(weight_lst)

        # a dictionary of {peptide_seq: averaged_hopp_score}
        pept_score_dict = {}

        # Calculate corrected score
        for i in range(0, len(resi_hopp_lst) - pep_length + 1):

            pept_score_lst = resi_hopp_lst[i:i + pep_length]
            weighted_pept_score_lst = []

            for score, weight in zip(pept_score_lst, weight_lst):
                weighted_score = score * weight
                weighted_pept_score_lst.append(weighted_score)

            pept_score = sum(weighted_pept_score_lst) / (sum(weight_lst))  # sum of scores averaged over sum of weights
            pept_seq = "".join(aa_lst[i:i + pep_length])
            pept_score_dict[pept_seq] = pept_score

        # key:value pair was switched in the turple to allow sorting by hopp score
        return [(v, k) for k, v in pept_score_dict.items()]


    # ## Examples of Usage

    # ### Example 1: Compute Hopp-Woods Scores Without Weights (window=7, $\alpha=1$)

    # In[2]:

    protein = st.text_input("Enter Protein Sequence ")

    # calculate averaged Hopp score
    if len(protein) > 0:
        result = calc_hopp(protein, 7)

        # print averaged Hopp score result, from lowest to highest
        st.write("(Avg Hopp Score Sorted, Peptide)")
        # for i in sorted(result, reverse=True):
        #     print ("{:.2f}".format(i[0]), "{}".format(i[1]))
        result_1 = sorted(result, reverse=True)
        st.dataframe(result_1)
        # Plot desired range to show on the x axis.
        # Recommend to change starting position to 1 instead of 0
        x = range(1, 24)

        # range of averaged hopp scores to show on y axis.
        y = [x[0] for x in result[0:23]]

        # plot chart
        plt.plot(x, y, "r-", x, y, "ro")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophilicity Score")

        # ### Example 2: Computed Hopp-Woods Scores Weighted by Linear Variation Model (window=7, $\alpha=0.1$)
        #
        # Same protein and window

        # In[3]:

        result_corr = calc_hopp(protein, 7, alpha=0.1)

        st.write("(Avg Hopp Score Sorted, Peptide)")
        # for i in sorted(result_corr, reverse=True):
        #     print ("{:.2f}".format(i[0]), "{}".format(i[1]))
        result_2 = sorted(result_corr, reverse=True)
        st.dataframe(result_2)
        y2 = [x[0] for x in result_corr[0:23]]
        plt.plot(x, y2, "r-", x, y2, "ro")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophilicity Score")

        # ### Example 3: Computed Hopp-Woods Scores Weighted by Linear Variation Model (window=10, $\alpha=0.1$)
        #
        # Same protein

        # In[4]:

        result_corr_2 = calc_hopp(protein, 10, alpha=0.1)

        st.write("(Avg Hopp Score Sorted, Peptide)")
        # for i in sorted(result_corr_2, reverse=True):
        #     print ("{:.2f}".format(i[0]), "{}".format(i[1]))
        result_3 = sorted(result_corr_2, reverse=True)
        st.dataframe(result_3)
        y3 = [x[0] for x in result_corr_2[0:23]]
        plt.plot(x, y3, "r-", x, y3, "ro")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophilicity Score")
        plt.savefig("234")
        st.image("234.png")
        # ## Validating against Expasy Result

        # ### Validate Example 1 (no weights, window=7)

        # In[6]:

        # list only the first 23 in the order of the sequence
        y_expassy = [-0.086, 0.414, 0.086, -0.300, 0.271, 0.271, -0.014, -0.300,
                     -0.800, -0.543, -0.329, -1.014, -1.057, -0.943, -0.657,
                     -0.843, -0.343, -0.343, -0.043, -0.000, 0.171, 0.086, 0.343,
                     ]

        # plt.figure(figsize=(12,6))
        # plt.plot(x, y, "r-", linewidth=7, alpha=0.4)
        # plt.plot(x, y_expassy, "b--")
        # plt.title("Comparison of Script Result vs Expasy Result (No Weights, Window=7)")
        # plt.xlabel("Amino Acid Position")
        # plt.ylabel("Hydrophilicity Score of Peptide")
        # plt.legend(["Script", "Expasy"], loc="lower right")

        # plt.savefig("expassy_validate_noweights.png", dpi=300)

        # ### Validate Example 2 (Weighted $\alpha=0.1$, window=7)

        # In[7]:

        y2_expassy = [-0.176, 0.059, 0.335, 0.379, 0.276, -0.182, -0.250, -0.018,
                      -0.253, -0.632, -0.968, -0.994, -0.932, -0.909, -0.921, -0.738,
                      -0.618, -0.247, -0.097, 0.344, 0.221, 0.256, 0.115
                      ]
        #
        # plt.figure(figsize=(12,6))
        # plt.plot(x, y2, "r-", linewidth=7, alpha=0.4)
        # plt.plot(x, y2_expassy, "b--")
        # plt.title("Comparison of Script Result vs Expasy Result (alpha=0.1, Window=7)")
        # plt.xlabel("Amino Acid Position")
        # plt.ylabel("Hydrophilicity Score of Peptide")
        # plt.legend(["Script", "Expasy"], loc="lower right")

        # plt.savefig("expassy_validate_weighted.png", dpi=300)
        st.write("If the positive score is >80% then the entered protein is antigenic otherwise non-antigenic.")




