import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import tree
import streamlit as st

from function import train_model

def app(df, x, y):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Visualisasi")

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x,y)
        plt.figure(figsize=(10,6))
        confusion_matrix(model, x, y, values_format='d')
        st.pyplot()

    if st.checkbox("Decision Tree"):
        model, score = train_model(x,y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=5, out_file=None, rounded=True,
            feature_names=x.columns, class_names=['Tidak', 'Ya']
        )

        st.graphviz_chart(dot_data)