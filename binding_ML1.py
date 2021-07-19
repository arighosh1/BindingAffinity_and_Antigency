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
st.markdown(

    """
    <style>
    .reportview-container {
        background: url(https://github.com/arighosh1/BindingAffinity_and_Antigency/blob/main/icon.jpeg?raw=true)
    }
   .sidebar .sidebar-content {
        background: url("url_goes_here")
        background-repeat: no-repeat
    }
    </style>
    """,
    unsafe_allow_html=True
)


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
    # !/usr/bin/env python
    # coding: utf-8

    # In[51]:

    file = ''
    uploaded_files = st.file_uploader("Choose a Protein File To Check Its Antigenicity : ", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        file = uploaded_file.read()

    if file != '':
        X = ProteinAnalysis(str(file))


        alanine = X.get_amino_acids_percent()['A']
        valine = X.get_amino_acids_percent()['V']
        isoleucine = X.get_amino_acids_percent()['I']
        leucine = X.get_amino_acids_percent()['L']
        sec_struc = X.secondary_structure_fraction()
            # st.write("Secondary structure fraction = ",sec_struc[0])
        epsilon_prot = X.molar_extinction_coefficient()
            # st.write(epsilon_prot)

            # In[52]:

            # Aliphatic index = X(Ala) + a * X(Val) + b * ( X(Ile) + X(Leu) )
        Aliphatic_index = ((alanine + 2.9) * (valine + 3.9) * (isoleucine + leucine)) / 10

            # In[53]:

        st.write("Aliphatic Index Value Is : ",Aliphatic_index)

        # In[54]:

        if Aliphatic_index >= 0.1:
            st.write("The Protein is antigenic protein")
        else:
            st.write("The Protein is non-antigenic protein")

else:
    st.write("Input Your Sequence")





