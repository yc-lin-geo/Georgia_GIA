import streamlit as st
import matplotlib.pyplot as plt

cat = ["bored", "happy", "bored", "bored", "happy", "bored"]
dog = ["happy", "happy", "happy", "happy", "bored", "bored"]
activity = ["combing", "drinking", "feeding", "napping", "playing", "washing"]

width = st.sidebar.slider("plot width", 1, 25, 3)
height = st.sidebar.slider("plot height", 1, 25, 1)

fig, ax = plt.subplots(figsize=(width, height))
ax.plot(activity, dog, label="dog")
ax.plot(activity, cat, label="cat")
ax.legend()

st.pyplot(fig)
