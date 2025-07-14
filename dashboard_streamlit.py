import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("spotify_cleaned_dataset.csv")

st.title("Dashboard Analisis Spotify User Preference")

st.header("1. Eksplorasi Data (EDA)")
st.write(df.describe())

# Plot distribusi
st.subheader("Distribusi Fitur")
selected_feature = st.selectbox("Pilih fitur untuk histogram", df.columns[1:-1])
fig, ax = plt.subplots()
sns.histplot(df[selected_feature], kde=True, ax=ax)
st.pyplot(fig)

# Korelasi
st.header("2. Korelasi antar fitur")
fig, ax = plt.subplots()
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Regresi sederhana
st.header("3. Regresi Linier Sederhana")
feature = st.selectbox("Pilih fitur prediktor:", df.columns[1:-1])
X = df[[feature]]
y = df['user_preference']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

fig, ax = plt.subplots()
sns.scatterplot(x=X[feature], y=y, ax=ax)
ax.plot(X, y_pred, color='red')
ax.set_title(f"Regresi Linier: {feature} vs User Preference")
st.pyplot(fig)
st.write(f"Koefisien: {model.coef_[0]:.4f}")
st.write(f"Intercept: {model.intercept_:.4f}")
