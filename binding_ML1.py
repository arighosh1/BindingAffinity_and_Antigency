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
img=Image.open('icon.jpeg')
st.set_page_config(page_title='RATIONAL VACCINE DESIGN FOR VIRUS USING MACHINE LEARNING APPROACHES', page_icon =img, layout = 'wide', initial_sidebar_state = 'auto')
import os

basedir = os.path.dirname(os.path.abspath(__file__))
basedir+="/icon.jpg"
# st.markdown(
#
#     """
#     <style>
#     .reportview-container {
#         background: url(https://github.com/arighosh1/BindingAffinity_and_Antigency/blob/main/icon.jpeg?raw=true);
#         height: 100%;
#         background-position: center;
#         background-repeat: no-repeat;
#         background-size: cover;
#     }
#    .sidebar .sidebar-content {
#         background: url("url_goes_here")
#
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.sidebar.image(img)
st.sidebar.title("Chose From SlideBar")
value = st.sidebar.slider("slide to 0 for  Affinity and 1 for Antigency",0,1)



if value==0:
    filename = st.file_uploader("Choose a  File To Check Its Binding Affinity : ", accept_multiple_files=False)
    if filename != None:
        st.title("Uploaded File Data Is")

        genedata = pd.read_csv(filename)
        genedata.head(5)
        st.dataframe(genedata)

        x = genedata.iloc[:, 1].values
        y = genedata.iloc[:, 3].values
        train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.25,
                                                                                    random_state=42)

        st.title("Training Set Information.")

        st.write('Training Features Shape : ', train_features.shape)
        st.write('Training Labels Shape : ', train_labels.shape)
        st.write('Testing Features Shape : ', test_features.shape)
        st.write('Testing Labels Shape : ', test_labels.shape)

        st.title("Binding Affintiy.")
        st.write("binding Affinity of protein and ligand:", test_features)

        # sns.distplot(random.binomial(n=genedata.kjmol, p=0.5), hist=False, label='binomial')
        # sns.distplot(random.poisson(lam=test_features), hist=False, label='poisson')
        # st.write(plt.show())
elif value==1:
    import streamlit as st
    #!/usr/bin/env python
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
    hopp_scores={
        "R": 3,
        "D": 3,
        "E": 3,
        "K": 3,
        "S":0.3,
        "N":0.2,
        "Q":0.2,
        "G":0,
        "P":0,
        "T":-0.4,
        "A":-0.5,
        "H":-0.5,
        "C":-1,
        "M":-1.3,
        "V":-1.5,
        "I":-1.8,
        "L":-1.8,
        "Y":-2.3,
        "F":-2.5,
        "W":-3.4
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
        weight_lst=[]
        if pept_length%2!=0:
            for idx in range (0, pept_length):
                if idx<=pept_length//2:
                    weight=alpha+(1-alpha)*idx/(pept_length//2)
                    weight=round(weight, 2)
                    weight_lst.append(weight)
                else:
                    weight=1-(1-alpha)*(idx-pept_length//2)/(pept_length//2)
                    weight=round(weight, 2)
                    weight_lst.append(weight)
        else:
            for idx in range (0, pept_length):
                if idx<pept_length/2:
                    weight=alpha+(1-alpha)*idx/(pept_length/2-1)
                    weight=round(weight, 2)
                    weight_lst.append(weight)
                else:
                    weight=1-(1-alpha)*(idx-pept_length/2)/(pept_length/2-1)
                    weight=round(weight, 2)
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


        #Caculate un-corrected score
        aa_lst=list(seq)
        resi_hopp_lst=[hopp_scores[x] for x in aa_lst]


        #Caculate weights
        weight_lst=lm(pep_length, alpha)
        st.write("Weights used: ", end="")
        st.write(weight_lst)


        #a dictionary of {peptide_seq: averaged_hopp_score}
        pept_score_dict={}


        #Calculate corrected score
        for i in range (0, len(resi_hopp_lst)-pep_length+1):

            pept_score_lst=resi_hopp_lst[i:i+pep_length]
            weighted_pept_score_lst=[]

            for score, weight in zip(pept_score_lst, weight_lst):
                weighted_score=score*weight
                weighted_pept_score_lst.append(weighted_score)

            pept_score=sum(weighted_pept_score_lst)/(sum(weight_lst)) #sum of scores averaged over sum of weights
            pept_seq="".join(aa_lst[i:i+pep_length])
            pept_score_dict[pept_seq]=pept_score

        #key:value pair was switched in the turple to allow sorting by hopp score
        return [(v, k) for k, v in pept_score_dict.items()]


    # ## Examples of Usage

    # ### Example 1: Compute Hopp-Woods Scores Without Weights (window=7, $\alpha=1$)

    # In[2]:


    protein=st.text_input("Enter Protein file ")

    #calculate averaged Hopp score
    if len(protein)>0:
        protein
        result=calc_hopp(protein, 7)



        #print averaged Hopp score result, from lowest to highest
        st.write("(Avg Hopp Score Sorted, Peptide)")
        # for i in sorted(result, reverse=True):
        #     print ("{:.2f}".format(i[0]), "{}".format(i[1]))
        st.dataframe(result)
        # Plot desired range to show on the x axis.
        # Recommend to change starting position to 1 instead of 0
        x=range(1, 24)

        #range of averaged hopp scores to show on y axis.
        y=[x[0] for x in result[0:23]]

        #plot chart
        plt.plot(x, y, "r-", x, y, "ro")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophilicity Score")

        # ### Example 2: Computed Hopp-Woods Scores Weighted by Linear Variation Model (window=7, $\alpha=0.1$)
        #
        # Same protein and window

        # In[3]:


        result_corr=calc_hopp(protein, 7, alpha=0.1)

        st.write("(Avg Hopp Score Sorted, Peptide)")
        # for i in sorted(result_corr, reverse=True):
        #     print ("{:.2f}".format(i[0]), "{}".format(i[1]))
        result_corr=sorted(result_corr, reverse=True)
        st.dataframe(result_corr)
        y2=[x[0] for x in result_corr[0:23]]
        plt.plot(x, y2, "r-", x, y2, "ro")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophilicity Score")


        # ### Example 3: Computed Hopp-Woods Scores Weighted by Linear Variation Model (window=10, $\alpha=0.1$)
        #
        # Same protein

        # In[4]:


        result_corr_2=calc_hopp(protein, 10, alpha=0.1)

        st.write("(Avg Hopp Score Sorted, Peptide)")
        # for i in sorted(result_corr_2, reverse=True):
        #     print ("{:.2f}".format(i[0]), "{}".format(i[1]))
        result_corr_2=sorted(result_corr_2, reverse=True)
        st.dataframe(result_corr_2)
        y3=[x[0] for x in result_corr_2[0:23]]
        plt.plot(x, y3, "r-", x, y3, "ro")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophilicity Score")
        plt.savefig("234")
        st.image("234.png")
        # ## Validating against Expasy Result

        # ### Validate Example 1 (no weights, window=7)

        # In[6]:


        #list only the first 23 in the order of the sequence
        y_expassy=[-0.086, 0.414, 0.086, -0.300, 0.271, 0.271, -0.014, -0.300,
                     -0.800, -0.543, -0.329, -1.014,  -1.057 , -0.943, -0.657,
                      -0.843, -0.343, -0.343, -0.043, -0.000, 0.171, 0.086,0.343,
        ]

        # plt.figure(figsize=(12,6))
        # plt.plot(x, y, "r-", linewidth=7, alpha=0.4)
        # plt.plot(x, y_expassy, "b--")
        # plt.title("Comparison of Script Result vs Expasy Result (No Weights, Window=7)")
        # plt.xlabel("Amino Acid Position")
        # plt.ylabel("Hydrophilicity Score of Peptide")
        # plt.legend(["Script", "Expasy"], loc="lower right")


        #plt.savefig("expassy_validate_noweights.png", dpi=300)


        # ### Validate Example 2 (Weighted $\alpha=0.1$, window=7)

        # In[7]:


        y2_expassy=[-0.176,0.059, 0.335, 0.379, 0.276,-0.182,-0.250,-0.018,
        -0.253, -0.632, -0.968 , -0.994, -0.932, -0.909, -0.921, -0.738,
        -0.618, -0.247, -0.097, 0.344, 0.221, 0.256 , 0.115
        ]
        #
        # plt.figure(figsize=(12,6))
        # plt.plot(x, y2, "r-", linewidth=7, alpha=0.4)
        # plt.plot(x, y2_expassy, "b--")
        # plt.title("Comparison of Script Result vs Expasy Result (alpha=0.1, Window=7)")
        # plt.xlabel("Amino Acid Position")
        # plt.ylabel("Hydrophilicity Score of Peptide")
        # plt.legend(["Script", "Expasy"], loc="lower right")

        #plt.savefig("expassy_validate_weighted.png", dpi=300)






