import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


st.title('My plotting app')
n=1
opn = st.sidebar.slider(
     'period(s):',
      n,1,10)

x = np.linspace(0,opn*2*3.14/n,100)

fig, ax = plt.subplots()
ax.plot(x, np.sin(x),linestyle=":",color="k",marker="x")

st.pyplot(fig)

# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig, clear_figure=False)

# st.write("Plotting is awesome")

# y = np.rand.rand(100)
# x = np.linspace(0,10,100)

# chart_data = pd.DataFrame(
#      x,y,
#      columns=['x','y'])

# st.line_chart(chart_data)

# x = 0


st.write(n)