import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import streamlit as st

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
    st.title("Working on code.")




