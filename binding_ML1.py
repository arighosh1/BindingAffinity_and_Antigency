


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
img = Image.open('icon1.jpg')
st.set_page_config(page_title='RATIONAL VACCINE DESIGN FOR VIRUS USING MACHINE LEARNING APPROACHES', page_icon=img,layout='wide', initial_sidebar_state='auto')
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
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("style.css")
def color_and_font(str,color):
    return "<div> <span class='highlight "+color+"'><span class='bold'>"+str+"</span></span></div>"


st.sidebar.image("https://github.com/arighosh1/BindingAffinity_and_Antigency/blob/main/icon1.jpg?raw=true")
value=st.sidebar.slider("Slide to 0 for Introduction, 1 for Binding Affinity, 2 for Antigenicity and 3 for Covid_Autoimmune",0,3,0)
st.sidebar.write("A Project submitted to the Faculty of The College of Engineering and Computer Science in Partial Fulfillment of the conditions required for the Degree of Doctor of Philosophy")
st.sidebar.write("by")
st.sidebar.write("Aritra Ghosh")
if value==0:
    st.title("Binding Affinity and Antigenicity Prediction Tool (BAAPT)")
    st.write("Prediction tools for binding affinity and antigenicity are currently webserver-based and do not follow a single software design pattern or reference architecture. Therefore, developing universal models and architectures that may be used as templates for the building of these prediction tools is crucial. As a result, this study explains the Binding Affinity and Antigenicity Prediction Tool (BAAPT)'s concept and pattern, as well as how it is applied to various reverse vaccinology systems. ")
    st.title("FAQ")
    st.write("Q: How to prepare dataset for Binding Affinity?")
    st.write("🎈🎈🎈Dataset Create 1 and 2: https://github.com/arighosh1/BindingAffinity_and_Antigency")
    st.write("Q: How can we get the sample dataset for Binding Affinity, protein sequence for Antigenicity, and vaccine data for Autoimmunity prediction?")
    st.write("🎈Binding Affinity:")
    st.write("🎈🎈🎈Training and Testing Dataset (created by Developer): protein-ligand.csv and protein-ligand-test.csv https://github.com/arighosh1/BindingAffinity_and_Antigency")
    st.write("🎈🎈🎈Training and Testing Dataset (taken from published research paper): pdbbind-2016-train.csv and pdbbind-2016-test.csv Jones, D., Kim, H., Zhang, X., Zemla, A., Stevenson, G., Bennett, W. F. D., Kirshner, D., Wong, S. E., Lightstone, F. C., & Allen, J. E. (2021). Improved Protein-Ligand Binding Affinity Prediction with Structure-Based Deep Fusion Inference. Journal of Chemical Information and Modeling, 61(4), 1583–1592.") 
    st.write("🎈Antigenicity:")
    st.write("🎈🎈🎈Protein Sequence: https://www.ncbi.nlm.nih.gov/protein/YP_009724390.1?report=fasta")
    st.write("🎈🎈🎈AntiRNA Antibodies: https://www.uniprot.org/blast/uniprot/B20211004F248CABF64506F29A91F8037F07B67D10159C09")
    st.write("🎈Autoimmunity:")
    st.write("🎈🎈🎈VAERS Data: https://vaers.hhs.gov/data/datasets.html")
    st.write("🎈🎈🎈VAERS Data Use Guide: https://vaers.hhs.gov/docs/VAERSDataUseGuide_en_September2021.pdf")
elif value == 1:
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

    st.title("Binding Affinity")
    st.write("The strength of the binding interaction between a single biomolecule (e.g., protein or DNA) and its ligand/binding partner is referred to as binding affinity (e.g., drug or inhibitor). The equilibrium dissociation constant (KD), which is used to evaluate, and rank order the strengths of bimolecular interactions, is commonly used to measure and report binding affinity. The lower the KD value, the greater the ligand's affinity for its target. The higher the KD value, the weaker the attraction and binding of the target molecule and ligand.")
    st.write("Non-covalent intermolecular interactions such as hydrogen bonding, electrostatic interactions, hydrophobic and Van der Waals forces between the two molecules all influence binding affinity. Furthermore, the presence of other molecules may affect the binding affinity of a ligand to its target molecule.")
    st.image("before_method.jpg")
    st.write("Method: ")
    st.image("after_method.jpg")
    st.write("Algorithms:")
    st.write("1.	KNN (K — Nearest Neighbors) is one of many (supervised learning) algorithms used in data mining and machine learning; it is a classifier algorithm in which the learning is based on 'how similar' one data (a vector) is to another.")
    st.write("2.	The random forest, as the name implies, is made up of many individual decision trees that work together as an ensemble. Each individual tree in the random forest produces a class prediction, and the class with the most votes become the prediction of our model.")
    st.write("3.	The support vector machine algorithm's goal is to find a hyperplane in an N-dimensional space (N — the number of features) that clearly classifies the data points. There are numerous hyperplanes that could be used to separate the two classes of data points. Our goal is to find a plane with the greatest margin, i.e., the greatest distance between data points from both classes. Maximizing the margin distance provides some reinforcement, allowing future data points to be classified with greater certainty.")
    st.write("")



    # In[3]:
    my_1 = color_and_font("Upload File Containing PDB_ID(Training Set) : ","blue")
    st.markdown(my_1, unsafe_allow_html=True)
    file1 = st.file_uploader("", accept_multiple_files=False)
    my_1=color_and_font("Upload File Containing PDB_ID(Testing Set) : ","blue")
    st.markdown(my_1, unsafe_allow_html=True)
    file2 = st.file_uploader(" ", accept_multiple_files=False)
    if file1 != None and file2!=None :



        # !/usr/bin/env python
        # coding: utf-8
        df = pd.read_csv(file1)
        df2 = pd.read_csv(file2)


        # Read the data


        # print(df.head())
        # st.write("Data Head",df.head())

        # In[ ]:

        # In[ ]:

        # In[ ]:

        # In[ ]:

        # In[3]:
        # Traning Sets
        X = df.iloc[:, [1, 2]]
        Y = df2.iloc[:, 1]

        # In[9]:

        # In[4]:

        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=123456)

        # In[ ]:

        # In[ ]:

        # # Optimized parameters
        # ## max_features = 'auto'
        # ## n_estimators=100
        # ## random_state = 1234

        # In[5]:

        models_RF_train = {"RF": RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                                                       max_features='auto', max_leaf_nodes=None,
                                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                                       min_samples_leaf=1, min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0, n_estimators=100,
                                                       n_jobs=None, oob_score=False, random_state=1234,
                                                       verbose=0, warm_start=False)}

        # In[6]:

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
        st.title("Random Forest Predicted Values")
        scores_RF_train = pd.Series(scores).T
        scores_RF_train

        # In[10]:

        # Calculate statistics for test set (Core set) based on RF model
        scores = {}
        for m in models_RF_train:
            Y_pred_test_rf = models_RF_train[m].predict(X)
            scores[m + "_test_r2"] = r2_score(Y, Y_pred_test_rf)
            scores[m + "_rmse_test"] = sqrt(mean_squared_error(Y, Y_pred_test_rf))
            scores[m + "_mae_test"] = mean_absolute_error(Y, Y_pred_test_rf)
            scores[m + "_pcc_test"] = pearsonr(Y, Y_pred_test_rf)

        scores_RF_test = pd.Series(scores).T
        scores_RF_test

        # In[13]:

        # Save the test prediction result
        Pred_y = pd.DataFrame({'Y_pred_rf': Y_pred_test_rf})
        Exp_y = pd.DataFrame(Y)
        Prediction = pd.concat([Exp_y, Pred_y], axis=1)

        # In[14]:

        YV_array = np.array(Y_valid)
        YT_array = np.array(Y_train)
        XV_array = np.array(X_valid)
        XT_array = np.array(X_train)

        # In[15]:

        from sklearn.neighbors import KNeighborsRegressor

        knn_model = KNeighborsRegressor(n_neighbors=3)

        # In[16]:

        knn_model.fit(XT_array, YT_array)

        # In[17]:

        from sklearn.metrics import mean_squared_error
        from math import sqrt

        train_preds = knn_model.predict(XT_array)

        # In[18]:

        # print("KNN Predicted Values:", train_preds)
        st.title("KNN Predicted Values:")
        st.write( train_preds)

        # In[19]:

        mse = mean_squared_error(YT_array, train_preds)
        rmse = sqrt(mse)
        # print("RMSE_train KNN:", rmse)
        st.title("RMSE_train KNN:",rmse)

        # In[33]:

        from matplotlib.colors import ListedColormap
        import seaborn as sns

        # In[70]:

        #plt.plot(train_preds)
        st.line_chart(train_preds)
        # plt.savefig("Antigenic_Value")
        # st.image("Antigenic_Value.png")

        st.title("\nSVM Predicted Values\n")

        # if(df==new_df):
        #     st.write("Same")
        # else:
        #     st.write("not same")
        # In[6]:
        import streamlit as st
        import numpy as np
        import pandas as pd
        import numpy as np

        # print(df.head())

        # my_1 = color_and_font("Upload File Containing PDB_ID(Training Set) : ", "blue")
        # st.markdown(my_1, unsafe_allow_html=True)
        # svm_file_1 = st.file_uploader("  ", accept_multiple_files=False)
        # my_1 = color_and_font("Upload File Containing PDB_ID(Testing Set) : ", "blue")
        # st.markdown(my_1, unsafe_allow_html=True)
        # svm_file_2 = st.file_uploader("   ", accept_multiple_files=False)
        X = df.iloc[:, [1, 2]]
        Y = df2.iloc[:, 1]
        # print(X, Y)
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(X, Y)

        # print("Training data :", x_train.shape)
        # print("Test data :", x_test.shape)

        from sklearn.preprocessing import StandardScaler

        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.fit_transform(x_test)
        from sklearn.svm import SVC
        from sklearn import preprocessing

        lab_enc = preprocessing.LabelEncoder()
        train_y = lab_enc.fit_transform(y_train)
        classifier = SVC(kernel='linear')
        classifier.fit(x_train, train_y)
        y_pred = classifier.predict(x_test)
        st.write(y_pred)
        from sklearn import metrics

        test_y = lab_enc.fit_transform(y_test)
        # print(metrics.accuracy_score(test_y, y_pred))
        import matplotlib.pyplot as myplt

        myplt.scatter(x_test[:, 0], x_test[:, 1], c=test_y)
        myplt.savefig("foo", dpi=100)
        st.title("Chart")
        st.image("foo.png")


elif value == 2:
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


    # Amino Acid Scale Defined by Hopp-Woods's original paper
    parker_hydrophilic = {
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
        resi_hopp_lst = [parker_hydrophilic[x] for x in aa_lst]

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
    st.title("Antigenicity")
    st.write(
        "The location of continuous epitopes has been linked to properties of polypeptide chains such as hydrophilicity, flexibility, accessibility, turns, exposed surface, polarity, and antigenic propensity. This has resulted in a search for empirical rules that would allow the position of continuous epitopes to be predicted based on specific features of the protein sequence. The propensity scales for each of the 20 amino acids are used in all prediction calculations. Each scale is made up of 20 values that are assigned to each amino acid residue based on their relative proclivity to possess the property described by the scale.")
    st.write("Method:")
    st.write(
        "The amino acids in an interval of the chosen length, centered around residue I are considered when computing the score for a given residue i. In other words, for a window size n, the score for residue I was computed by subtracting the I - (n-1)/2 neighboring residues on each side of the residue. The score for residue I is the average of the scale values for these amino acids, unless otherwise specified (see table 1 for specific method implementation details). In general, a window size of 5 to 7 is appropriate for detecting potentially antigenic regions.")
    st.write("Hopp-Woods scale:")
    st.write(
        "Hopp-Woods scale: Hopp and Woods created their hydrophobicity scale to help them identify potentially antigenic sites in proteins. This scale is essentially a hydrophilic index with negative values assigned to apolar residues. When a window size of 7 is used, antigenic sites are more likely to be predicted.")
    st.write("Parker scale:")
    st.write(
        "In this method, a hydrophilic scale based on peptide retention times was constructed using high-performance liquid chromatography (HPLC) on a reversed-phase column. The epitope region was examined using a seven-residue window. The corresponding scale value was introduced for each of the seven residues, and the arithmetic mean of the seven residue values was assigned to the segment's fourth, (i+3), residue.")
    st.write("1.	Enter a protein sequence in plain format")
    st.write("2.	Select a prediction method")
    st.write("3.	Press Enter")

    # -*- coding: utf-8 -*-
    # """parkker.ipynb
    #
    # Automatically generated by Colaboratory.
    #
    # Original file is located at
    #     https://colab.research.google.com/drive/1o8ow33Ow1xiye5dx4R0huqiN4DFLHdQS
    # """

    import streamlit as st

    import matplotlib.pyplot as plt

    # Amino Acid Scale Defined by Hopp-Woods's original paper
    hopp_scores = {
        "R": 0.87,
        "D": 2.46,
        "E": 1.86,
        "K": 1.28,
        "S": 1.50,
        "N": 1.64,
        "Q": 1.37,
        "G": 1.28,
        "P": 0.30,
        "T": 1.15,
        "A": 0.03,
        "H": 0.30,
        "C": 0.11,
        "M": -1.41,
        "V": -1.27,
        "I": -2.45,
        "L": -2.78,
        "Y": -0.78,
        "F": -2.78,
        "W": -3.00
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
        print("Weights used: ", end="")
        print(weight_lst)
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


    # provide protein seq
    # protein = input("Enter Protein Sequence ")
    protein = st.text_input("Enter Protein Sequence ")

    protein_hopp=protein
    if len(protein) > 0:
        st.title("Parker")
        # calculate averaged Hopp score
        result = calc_hopp(protein, 7)


        # print averaged Hopp score result, from lowest to highest
        # print("(Avg Hopp Score Sorted, Peptide)")
        st.title("(Avg Parker Score Sorted, Peptide)")

        # st.write(avg_score)
        # for i in sorted(result, reverse=True):
        #     # st.write("{:.2f}".format(i[0]),"    ","'{}'".format(i[1]))
        avg_score = sorted(result, reverse=True)
        st.dataframe(avg_score)

        # Plot desired range to show on the x axis.
        # Recommend to change starting position to 1 instead of 0
        x = range(1, 24)

        # range of averaged hopp scores to show on y axis.
        y = [x[0] for x in result[0:23]]

        # plot chart
        st.title("Parker Graph")

        plt.figure(1)
        plt.plot(x, y, "r-", x, y, "ro")
        plt.xlabel("Amino Acid Position")
        plt .ylabel("Hydrophilicity Score")
        plt.savefig("parker",dpi=100)
        st.image("parker.png")
        st.title("Hopp Score")


        # hopp start
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


        def lm_hopp(pept_length, alpha):
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


        def calc_hopps(seq, pep_length, alpha=1):
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
            weight_lst = lm_hopp(pep_length, alpha)
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

                pept_score = sum(weighted_pept_score_lst) / (
                    sum(weight_lst))  # sum of scores averaged over sum of weights
                pept_seq = "".join(aa_lst[i:i + pep_length])
                pept_score_dict[pept_seq] = pept_score

            # key:value pair was switched in the turple to allow sorting by hopp score
            return [(v, k) for k, v in pept_score_dict.items()]


        # ## Examples of Usage

        # ### Example 1: Compute Hopp-Woods Scores Without Weights (window=7, $\alpha=1$)

        # In[2]:

        # calculate averaged Hopp score

        pep_length = st.slider("Window Size : ", 7, 9, 7)
        alpha = st.slider("Alpha value (/=10) : ", 1, 5, 1)
        alpha /= 10
        result = calc_hopps(protein_hopp, pep_length, alpha)

        # print averaged Hopp score result, from lowest to highest
        st.title("(Avg Hopp Score Sorted, Peptide)")

        result_1 = sorted(result, reverse=True)
        st.dataframe(result_1)
        # Plot desired range to show on the x axis.
        # Recommend to change starting position to 1 instead of 0
        x = range(1, 24)

        # range of averaged hopp scores to show on y axis.
        y = [x[0] for x in result[0:23]]

        # plot chart
        st.title("Hopp Graph")
        plt.figure(2)
        plt.plot(x, y, "r-", x, y, "ro")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophilicity Score")
        plt.savefig("hopp", dpi=100)
        st.write("Window Size = ", pep_length)
        st.write("Alpha Value = ", alpha)
        st.image("hopp.png")

        # ## Validating against Expasy Result

        # In[7]:

        y_expassy = [-0.176, 0.059, 0.335, 0.379, 0.276, -0.182, -0.250, -0.018,
                      -0.253, -0.632, -0.968, -0.994, -0.932, -0.909, -0.921, -0.738,
                      -0.618, -0.247, -0.097, 0.344, 0.221, 0.256, 0.115
                      ]



elif value == 3:


    import pandas as pd
    import numpy as np
    from pymatch.Matcher import Matcher
    import re
    from tqdm.autonotebook import tqdm
    import warnings
    from IPython.testing.globalipapp import get_ipython
    import streamlit as st
    st.title("Autoimmunity Prediction for COVID-19")
    st.write("This study looks at the long-term consequences of immunization. It investigates a group of unfavorable occurrences known as 'autoimmune diseases,' which involve the immune system attacking its own cells and include LLD, rheumatoid arthritis, and Crohn's disease. The following types of adverse occurrences are considered. Autoimmunity is a serious issue for COVID-19 vaccination recipients. Autoimmune illnesses are unpleasant, long-lasting, and can reduce a person's earning potential. Certain autoimmune diseases can be deadly or severe in rare circumstances. We used the Vaccine Adversity Reporting System (VAERS) input data set (1990-2021). It's a countrywide early warning system for possible security concerns with American vaccinations. The results demonstrate that the COVID-19 vaccination is safe for those with autoimmune diseases. When a person reports a safety signal, the risk of receiving a COVID-19 vaccination is only twice as high as when a person receives a non-COVID-19 vaccine. It's important to note that the link isn't causative. These investigations do not establish a cause; rather, they aid in the identification of possible issues that may be investigated further using other approaches.")

    tqdm.pandas()


    warnings.filterwarnings('ignore')

    ip=get_ipython()
    ip.run_line_magic('matplotlib', 'inline')

    # Fixed seed for reproducibility
    np.random.seed(4072021)

    # print(f"Python version: {sys.version}")
    # print(f"OS version: {platform.platform()}")
    # print(f"pandas version: {pd.__version__}")
    # print(f"numpy version: {np.__version__}")
    # print(f"scipy version: {scipy.__version__}")
    # print(f"statsmodels version: {statsmodels.__version__}")


    # In[4]:

    vac = st.file_uploader("Enter your (Vaccine information (e.g., vaccine name, manufacturer, lot number, route, site, and number of previous doses administered)) : ")
    rec = st.file_uploader("Enter your (Other vaccine information) : ")
    sym = st.file_uploader("Enter your (Symptoms information) : ")
    st.write("🎈Remember that positive predictive value (PPV) varies with disease prevalence when interpreting results from diagnostic tests. Prevalence is a measure of how common a disease is in a specified at-risk population at a specific time point or period. PPV is the percent of positive test results that are true positives. As disease prevalence decreases, the percent of test results that are false positives increase.")
    st.write("🎈For example, a test with 98% specificity would have a PPV of just over 80% in a population with 10% prevalence, meaning 20 out of 100 positive results would be false positives.")
    st.write("🎈The same test would only have a PPV of approximately 30% in a population with 1% prevalence, meaning 70 out of 100 positive results would be false positives.  This means that, in a population with 1% prevalence, only 30% of individuals with positive test results actually have the disease.")
    st.write("🎈At 0.1% prevalence, the PPV would only be 4%, meaning that 96 out of 100 positive results would be false positives.")
    st.write("🎈Health care providers should take the local prevalence into consideration when interpreting diagnostic test results.")
    st.write("🎈Consider positive results in combination with clinical observations, patient history, and epidemiological information.")
    st.write("🎈A false negative is a test result that is wrong, because it indicates the person is not infected when they really are, or that they don’t have antibodies when they actually do.")
    st.write("🎈A false positive is a test result that is wrong, because it indicates the person is infected when they really are not or that they have antibodies when they really don’t.")
    if(vac!=None and rec!=None and sym!=None):
        vax_frames = []

        df = pd.read_csv(vac, index_col=None, header=0, encoding="latin")
        vax_frames.append(df)

        vax = pd.concat(vax_frames, axis=0, ignore_index=True)[["VAERS_ID", "VAX_TYPE"]]
        vax["VAX_TYPE"] = vax["VAX_TYPE"] == "COVID19"
        vax.columns = ["VAERS_ID", "IS_COVID_VACCINE"]


        # In[3]:




        recipient_frames = []

        df = pd.read_csv(rec, index_col=None, header=0, encoding="latin")
        recipient_frames.append(df)

        recipients = pd.concat(recipient_frames, axis=0, ignore_index=True)[["VAERS_ID", "SEX", "CAGE_YR"]]


        # In[5]:


        age_bands = {0: "<18",
                     18: "18-25",
                     26: "26-40",
                     41: "41-55",
                     56: "56-70",
                     71: ">70",
                     int(recipients.CAGE_YR.max()): "max"}

        recipients["AGE"] = pd.cut(recipients.CAGE_YR, bins=list(age_bands.keys()), labels=list(age_bands.values())[:-1])
        recipients = recipients.drop("CAGE_YR", axis=1).dropna()


        # In[6]:



        symptoms_frames = []

        df = pd.read_csv(sym, index_col=None, header=0)
        symptoms_frames.append(df)


        symptoms = pd.melt(pd.concat(symptoms_frames, axis=0, ignore_index=True)[["VAERS_ID", "SYMPTOM1", "SYMPTOM2", "SYMPTOM3", "SYMPTOM4", "SYMPTOM5"]],
                       id_vars="VAERS_ID",
                       value_vars=(f"SYMPTOM{i}" for i in range(1, 6))).drop("variable", axis=1)

        symptoms.columns = ("VAERS_ID", "SYMPTOM")


        # In[7]:


        vaccination_data = vax.merge(recipients, how="inner", on="VAERS_ID")


        # In[8]:


        autoimmune_conditions = (
            "Alveolar proteinosis",
            "Ankylosing spondylitis",
            "Anti-glomerular basement membrane disease",
            "Antisynthetase syndrome",
            "Autoimmune colitis",
            "Autoimmune disorder",
            "Autoimmune enteropathy",
            "Autoimmune eye disorder",
            "Autoimmune hyperlipidaemia",
            "Autoimmune inner ear disease",
            "Autoimmune lung disease",
            "Autoimmune lymphoproliferative syndrome",
            "Autoimmune myocarditis",
            "Autoimmune nephritis",
            "Autoimmune pericarditis",
            "Autoimmune retinopathy",
            "Autoimmune uveitis",
            "Axial spondyloarthritis",
            "Birdshot chorioretinopathy",
            "Chronic autoimmune glomerulonephritis",
            "Chronic gastritis",
            "Chronic recurrent multifocal osteomyelitis",
            "Coeliac disease",
            "Collagen disorder",
            "Collagen-vascular disease",
            "Cryofibrinogenaemia",
            "Cryoglobulinaemia",
            "Dermatomyositis",
            "Dressler's syndrome",
            "Glomerulonephritis rapidly progressive",
            "Goodpasture's syndrome",
            "Immunoglobulin G4 related disease",
            "IPEX syndrome",
            "Juvenile spondyloarthritis",
            "Keratoderma blenorrhagica",
            "Mixed connective tissue disease",
            "Myocarditis post infection",
            "Ocular pemphigoid",
            "Ocular pemphigoid",
            "Overlap syndrome",
            "Polychondritis",
            "Postpericardiotomy syndrome",
            "Pulmonary renal syndrome",
            "Satoyoshi syndrome",
            "Sjogren's syndrome",
            "Sympathetic ophthalmia",
            "Testicular autoimmunity",
            "Undifferentiated connective tissue disease",
            "Antiphospholipid syndrome",
            "Autoimmune anaemia",
            "Autoimmune aplastic anaemia",
            "Autoimmune haemolytic anaemia",
            "Autoimmune heparin-induced thrombocytopenia",
            "Autoimmune neutropenia",
            "Autoimmune pancytopenia",
            "Cold type haemolytic anaemia",
            "Coombs positive haemolytic anaemia",
            "Evans syndrome",
            "Pernicious anaemia",
            "Warm type haemolytic anaemia",
            "Addison's disease",
            "Atrophic thyroiditis",
            "Autoimmune endocrine disorder",
            "Autoimmune hypothyroidism",
            "Autoimmune pancreatitis",
            "Autoimmune thyroid disorder",
            "Autoimmune thyroiditis",
            "Basedow's disease",
            "Diabetic mastopathy",
            "Endocrine ophthalmopathy",
            "Hashimoto's encephalopathy",
            "Hashitoxicosis",
            "Insulin autoimmune syndrome",
            "Ketosis-prone diabetes mellitus",
            "Latent autoimmune diabetes in adults",
            "Lymphocytic hypophysitis",
            "Marine Lenhart syndrome",
            "Polyglandular autoimmune syndrome type I",
            "Polyglandular autoimmune syndrome type II",
            "Polyglandular autoimmune syndrome type III",
            "Silent thyroiditis",
            "Type 1 diabetes mellitus",
            "Alloimmune hepatitis",
            "Autoimmune cholangitis",
            "Autoimmune hepatitis",
            "Cholangitis sclerosing",
            "Primary biliary cholangitis",
            "Acute cutaneous lupus erythematosus",
            "Butterfly rash",
            "Central nervous system lupus",
            "Chronic cutaneous lupus erythematosus",
            "Cutaneous lupus erythematosus",
            "Lupoid hepatic cirrhosis",
            "Lupus cystitis",
            "Lupus encephalitis",
            "Lupus endocarditis",
            "Lupus enteritis",
            "Lupus hepatitis",
            "Lupus myocarditis",
            "Lupus myositis",
            "Lupus nephritis",
            "Lupus pancreatitis",
            "Lupus pleurisy",
            "Lupus pneumonitis",
            "Lupus vasculitis",
            "Lupus-like syndrome",
            "Neonatal lupus erythematosus",
            "Neuropsychiatric lupus",
            "Pericarditis lupus",
            "Peritonitis lupus",
            "Shrinking lung syndrome",
            "SLE arthritis",
            "Subacute cutaneous lupus erythematosus",
            "Systemic lupus erythematosus",
            "Systemic lupus erythematosus rash",
            "Autoimmune myositis",
            "Congenital myasthenic syndrome",
            "Inclusion body myositis",
            "Juvenile polymyositis",
            "Morvan syndrome",
            "Myasthenia gravis",
            "Myasthenia gravis crisis",
            "Myasthenia gravis neonatal",
            "Myasthenic syndrome",
            "Neuromyotonia",
            "Ocular myasthenia",
            "Polymyalgia rheumatica",
            "Polymyositis",
            "Acute disseminated encephalomyelitis",
            "Acute haemorrhagic leukoencephalitis",
            "Acute motor axonal neuropathy",
            "Acute motor-sensory axonal neuropathy",
            "Anti-myelin-associated glycoprotein associated polyneuropathy",
            "Autoimmune demyelinating disease",
            "Autoimmune encephalopathy",
            "Autoimmune neuropathy",
            "Axonal and demyelinating polyneuropathy",
            "Bickerstaff's encephalitis",
            "Chronic inflammatory demyelinating polyradiculoneuropathy",
            "Chronic lymphocytic inflammation with pontine perivascular enhancement responsive to steroid",
            "Concentric sclerosis",
            "Demyelinating polyneuropathy",
            "Encephalitis allergic",
            "Encephalitis autoimmune",
            "Faciobrachial dystonic seizure",
            "Guillain-Barre syndrome",
            "Leukoencephalomyelitis",
            "Limbic encephalitis",
            "Multiple sclerosis",
            "Myelitis transverse",
            "Neuralgic amyotrophy",
            "Neuromyelitis optica pseudo relapse",
            "Neuromyelitis optica spectrum disorder",
            "Paediatric autoimmune neuropsychiatric disorders associated with streptococcal infection",
            "POEMS syndrome",
            "Radiologically isolated syndrome",
            "Rasmussen encephalitis",
            "Secondary cerebellar degeneration",
            "Stiff leg syndrome",
            "Stiff person syndrome",
            "Subacute inflammatory demyelinating polyneuropathy",
            "Susac's syndrome",
            "Toxic oil syndrome",
            "Autoimmune arthritis",
            "Caplan's syndrome",
            "Cogan's syndrome",
            "Felty's syndrome",
            "Juvenile idiopathic arthritis",
            "Laryngeal rheumatoid arthritis",
            "Palindromic rheumatism",
            "Rheumatoid arthritis",
            "Rheumatoid lung",
            "Rheumatoid neutrophilic dermatosis",
            "Rheumatoid nodule",
            "Rheumatoid scleritis",
            "Rheumatoid vasculitis",
            "CREST syndrome",
            "Digital pitting scar",
            "Morphoea",
            "Reynold's syndrome",
            "Sclerodactylia",
            "Scleroderma",
            "Scleroderma associated digital ulcer",
            "Scleroderma renal crisis",
            "Scleroderma-like reaction",
            "Systemic scleroderma",
            "Systemic sclerosis pulmonary",
            "Acquired epidermolysis bullosa",
            "Alopecia areata",
            "Autoimmune blistering disease",
            "Autoimmune dermatitis",
            "Benign familial pemphigus",
            "Dermatitis herpetiformis",
            "Eosinophilic fasciitis",
            "Epidermolysis",
            "Granulomatous dermatitis",
            "Herpes gestationis",
            "Interstitial granulomatous dermatitis",
            "Linear IgA disease",
            "Nephrogenic systemic fibrosis",
            "Palisaded neutrophilic granulomatous dermatitis",
            "Paraneoplastic dermatomyositis",
            "Paraneoplastic pemphigus",
            "Pemphigoid",
            "Pemphigus",
            "Pityriasis lichenoides et varioliformis acuta",
            "Progressive facial hemiatrophy",
            "Pyoderma gangrenosum",
            "Vitiligo",
            "Autoinflammation with infantile enterocolitis",
            "Autoinflammatory disease",
            "Blau syndrome",
            "CANDLE syndrome",
            "Chronic infantile neurological cutaneous and articular syndrome",
            "Cryopyrin associated periodic syndrome",
            "Deficiency of the interleukin-1 receptor antagonist",
            "Deficiency of the interleukin-36 receptor antagonist",
            "Familial cold autoinflammatory syndrome",
            "Hyper IgD syndrome",
            "Majeed's syndrome",
            "Mevalonate kinase deficiency",
            "Mevalonic aciduria",
            "Muckle-Wells syndrome",
            "PASH syndrome",
            "PSTPIP1-associated myeloid-related proteinaemia inflammatory syndrome",
            "Pyogenic sterile arthritis pyoderma gangrenosum and acne syndrome",
            "Still's disease",
            "Acquired amegakaryocytic thrombocytopenia",
            "Acquired complement deficiency disease",
            "Acute graft versus host disease",
            "Acute graft versus host disease in intestine",
            "Acute graft versus host disease in liver",
            "Acute graft versus host disease in skin",
            "Acute graft versus host disease oral",
            "Amegakaryocytic thrombocytopenia",
            "Anamnestic reaction",
            "Aplasia pure red cell",
            "Arthritis enteropathic",
            "Arthritis reactive",
            "Bacille Calmette-Guerin scar reactivation",
            "Bronchiolitis obliterans syndrome",
            "C1q nephropathy",
            "C3 glomerulopathy",
            "CEC syndrome",
            "Central nervous system immune reconstitution inflammatory response",
            "Chronic graft versus host disease",
            "Chronic graft versus host disease in eye",
            "Chronic graft versus host disease in intestine",
            "Chronic graft versus host disease in liver",
            "Chronic graft versus host disease in skin",
            "Chronic graft versus host disease oral",
            "Colitis ulcerative",
            "Congenital thrombocytopenia",
            "Crohn's disease",
            "Cystitis interstitial",
            "Cytokine release syndrome",
            "Cytokine storm",
            "Cytophagic histiocytic panniculitis",
            "De novo purine synthesis inhibitors associated acute inflammatory syndrome",
            "Decreased immune responsiveness",
            "Encephalitis post varicella",
            "Engraftment syndrome",
            "Enteropathic spondylitis",
            "Episcleritis",
            "Erythema nodosum",
            "Erythrodermic psoriasis",
            "Febrile infection-related epilepsy syndrome",
            "Fibrillary glomerulonephritis",
            "Giant cell myocarditis",
            "Glomerulonephritis",
            "Graft versus host disease",
            "Graft versus host disease in eye",
            "Graft versus host disease in gastrointestinal tract",
            "Graft versus host disease in liver",
            "Graft versus host disease in lung",
            "Graft versus host disease in skin",
            "Guttate psoriasis",
            "Haemophagocytic lymphohistiocytosis",
            "Heparin-induced thrombocytopenia",
            "Hypergammaglobulinaemia",
            "Hypocomplementaemia",
            "Idiopathic interstitial pneumonia",
            "Idiopathic pulmonary fibrosis",
            "IgA nephropathy",
            "IgM nephropathy",
            "Immune reconstitution inflammatory syndrome",
            "Immune reconstitution inflammatory syndrome associated Kaposi's sarcoma",
            "Immune reconstitution inflammatory syndrome associated tuberculosis",
            "Immune recovery uveitis",
            "Immune system disorder",
            "Immune thrombocytopenia",
            "Immune-mediated adverse reaction",
            "Immune-mediated arthritis",
            "Immune-mediated cholangitis",
            "Immune-mediated cholestasis",
            "Immune-mediated cytopenia",
            "Immune-mediated dermatitis",
            "Immune-mediated encephalitis",
            "Immune-mediated encephalopathy",
            "Immune-mediated endocrinopathy",
            "Immune-mediated enterocolitis",
            "Immune-mediated gastritis",
            "Immune-mediated hepatic disorder",
            "Immune-mediated hepatitis",
            "Immune-mediated hyperthyroidism",
            "Immune-mediated hypothyroidism",
            "Immune-mediated myocarditis",
            "Immune-mediated myositis",
            "Immune-mediated nephritis",
            "Immune-mediated neuropathy",
            "Immune-mediated pancreatitis",
            "Immune-mediated pneumonitis",
            "Immune-mediated renal disorder",
            "Immune-mediated thyroiditis",
            "Immune-mediated uveitis",
            "Immunisation reaction",
            "Infection masked",
            "Infection susceptibility increased",
            "Interstitial lung disease",
            "Iritis",
            "Jarisch-Herxheimer reaction",
            "Juvenile psoriatic arthritis",
            "Kaposi sarcoma inflammatory cytokine syndrome",
            "Keratic precipitates",
            "Lewis-Sumner syndrome",
            "Mast cell activation syndrome",
            "Mastocytic enterocolitis",
            "Mazzotti reaction",
            "Membranous-like glomerulopathy with masked IgG-kappa deposits",
            "Metastatic cutaneous Crohn's disease",
            "Multifocal motor neuropathy",
            "Multiple chemical sensitivity",
            "Multisystem inflammatory syndrome in children",
            "Myofascitis",
            "Nail psoriasis",
            "Neonatal alloimmune thrombocytopenia",
            "Neonatal Crohn's disease",
            "Neuronophagia",
            "Neutrophil extracellular trap formation",
            "Noninfectious myelitis",
            "Obliterative bronchiolitis",
            "Optic neuritis",
            "Optic perineuritis",
            "Palmoplantar pustulosis",
            "Paradoxical psoriasis",
            "Paraneoplastic retinopathy",
            "Pathergy reaction",
            "Pleuroparenchymal fibroelastosis",
            "Polyneuropathy in malignant disease",
            "Postcolectomy panenteritis",
            "Pre-engraftment immune reaction",
            "Proctitis ulcerative",
            "Pseudomononucleosis",
            "Psoriasis",
            "Psoriatic arthropathy",
            "Pulmonary sensitisation",
            "Pustular psoriasis",
            "Pustulotic arthro-osteitis",
            "Pyostomatitis vegetans",
            "Reactive angioendotheliomatosis",
            "Reactive capillary endothelial proliferation",
            "Rebound psoriasis",
            "Retroperitoneal fibrosis",
            "Rheumatic brain disease",
            "Rheumatic disorder",
            "Rheumatic fever",
            "Sacroiliitis",
            "SAPHO syndrome",
            "Scleritis",
            "Sensitisation",
            "Sepsis syndrome",
            "Spontaneous heparin-induced thrombocytopenia syndrome",
            "Subacute sclerosing panencephalitis",
            "Systemic immune activation",
            "Systemic mastocytosis",
            "Tachyphylaxis",
            "Thymus disorder",
            "Thymus enlargement",
            "Transfusion associated graft versus host disease",
            "Transfusion microchimerism",
            "Tubulointerstitial nephritis and uveitis syndrome",
            "Ulcerative keratitis",
            "Uveitis",
            "Vogt-Koyanagi-Harada disease",
            "Acute haemorrhagic oedema of infancy",
            "Administration site vasculitis",
            "Angiopathic neuropathy",
            "Anti-neutrophil cytoplasmic antibody positive vasculitis",
            "Application site vasculitis",
            "Behcet's syndrome",
            "Catheter site vasculitis",
            "Central nervous system vasculitis",
            "Cutaneous vasculitis",
            "Diffuse vasculitis",
            "Eosinophilic granulomatosis with polyangiitis",
            "Giant cell arteritis",
            "Granulomatosis with polyangiitis",
            "Haemorrhagic vasculitis",
            "Henoch-Schonlein purpura",
            "Henoch-Schonlein purpura nephritis",
            "Hypersensitivity vasculitis",
            "Infected vasculitis",
            "Infusion site vasculitis",
            "Injection site vasculitis",
            "Kawasaki's disease",
            "MAGIC syndrome",
            "Medical device site vasculitis",
            "Microscopic polyangiitis",
            "Nodular vasculitis",
            "Ocular vasculitis",
            "Palpable purpura",
            "Polyarteritis nodosa",
            "Pseudovasculitis",
            "Pulmonary vasculitis",
            "Renal vasculitis",
            "Retinal vasculitis",
            "Segmented hyalinising vasculitis",
            "Stoma site vasculitis",
            "Takayasu's arteritis",
            "Thromboangiitis obliterans",
            "Vaccination site vasculitis",
            "Vasculitic rash",
            "Vasculitic ulcer",
            "Vasculitis",
            "Vasculitis gastrointestinal",
            "Vasculitis necrotising"
        )


        # In[9]:


        p_normals = r".*negative$|.*\snormal$|.*(scopy|graphy|gram|metry|opsy)$|.*(count|percentage|level|test|assay|culture|X-ray|imaging|gradient|band(s)?|index|surface area|gas|scale|antibod(y|ies)|urine absent|Carotid pulse|partial pressure|time|P(C)?O2)$|Oxygen saturation$|End-tidal.*"
        p_tests = r".*(ase|ose|ine|enzyme|in|ine|ines|ium|ol|ole|ate|lytes|ogen|gases|oids|ide|one|an|copper|iron)$|.*(level therapeutic)$|.*(globulin)\s.{1,2}$|Barium (swallow|enema)"
        p_procedures = r".*(plasty|insertion|tomy|ery|puncture|therapy|treatment|tripsy|operation|repair|procedure|bypass|insertion|removal|graft|closure|implant|lavage|support|transplant|match|bridement|application|ablation)$|Incisional drainage$|.* stimulation$|Immunisation$"
        p_normal_procedures = r"(Biopsy|pH|.* examination|X-ray|.* pulse|Blood|Electro(.*)gram|.* test(s)?|Echo(.*)gram|.*(scopy)|Cardiac (imaging|monitoring|ventriculogram)|Chromosomal|Carbohydrate antigen|Cell marker|.* examination|Computerised tomogram|Culture|.* evoked potential(s)?|Cytology|Doppler)(?!.*(abnormal|increased|decreased|depression|elevation|present|absent))"
        p_managements = r"(Catheter|Device\).*|.* care$|.* user$|Cardiac pacemaker .*"
        p_other_irrelevants = r"Blood group.*|Blood don(or|ation)$|Drug (abuse(r)?|dependence|screen).*|Elderly|Non-tobacco user|No adverse event"
        p_covid_related = r".*COVID-19(prophylaxis|immunisation|screening)|Asymptomatic COVID-19"

        p = re.compile("|".join([p_normals, p_tests, p_procedures, p_normal_procedures, p_other_irrelevants, p_covid_related]))


        # In[10]:


        symptoms = symptoms[symptoms.SYMPTOM.str.match(p) == False]


        # In[11]:


        symptoms["IS_AUTOIMMUNE"] = symptoms.SYMPTOM.isin(autoimmune_conditions)


        # In[12]:


        instances = vax.merge(symptoms[["VAERS_ID", "IS_AUTOIMMUNE"]].groupby("VAERS_ID").agg({"IS_AUTOIMMUNE": np.max}), how="inner", left_on="VAERS_ID", right_index=True).merge(recipients, how="inner").groupby("VAERS_ID").first()

        # In[13]:


        maj_min_value=pd.crosstab(instances.IS_COVID_VACCINE, instances.IS_AUTOIMMUNE)
        st.title(maj_min_value)
        st.write(maj_min_value)


        # In[14]:


        cases = instances[instances.IS_AUTOIMMUNE == True]
        controls = instances[instances.IS_AUTOIMMUNE == False]

        cases["VAERS_ID"] = cases.index
        controls["VAERS_ID"] = controls.index

        # In[15]:

        m = Matcher(cases, controls, yvar="IS_AUTOIMMUNE", exclude=["VAERS_ID"])

        # In[ ]:






