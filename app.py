import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

# --- SETUP DASHBOARD ---
st.set_page_config(layout="centered")
st.title(" Dashboard Analisis Regresi pada Dataset Spotify")

# --- LOAD DATA ---
df = pd.read_csv("spotify_cleaned_dataset.csv")

# --- TAB 1: EDA ---
st.header("1. Exploratory Data Analysis (EDA)")
st.subheader(" Ringkasan Data")
st.write(df.describe())

st.subheader(" Cek Missing Values")
st.write(df.isnull().sum())

st.subheader(" Distribusi Fitur Numerik")
fig, ax = plt.subplots(figsize=(12, 6))
df.drop(columns=['track_name']).hist(bins=15, layout=(2, 3), ax=ax)
st.pyplot(fig)

# --- TAB 2: Korelasi Heatmap ---
st.header("2. Korelasi Fitur")
correlation = df.drop(columns=['track_name']).corr()

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# --- TAB 3: Regresi Linier Sederhana ---
st.header("3. Regresi Linier Sederhana")
features = [col for col in df.columns if col not in ['track_name', 'user_preference']]
n_features = len(features)
cols = 3
rows = int(np.ceil(n_features / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
axes = axes.flatten()

results = []

for i, feature in enumerate(features):
    X = df[[feature]]
    y = df['user_preference']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Plot
    sns.scatterplot(x=X[feature], y=y, ax=axes[i])
    axes[i].plot(X, y_pred, color='red')
    axes[i].set_title(f"{feature}\nR² = {r2:.2f}")

    # Simpan hasil
    results.append({
        'Feature': feature,
        'Koefisien': model.coef_[0],
        'Intercept': model.intercept_,
        'R²': r2
    })

# Hapus subplot kosong
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
st.pyplot(fig)

# Tabel hasil regresi sederhana
st.dataframe(pd.DataFrame(results).sort_values(by="R²", ascending=False), use_container_width=True)

# --- TAB 4: Regresi Linier Berganda ---
st.header("4. Regresi Linier Berganda")

X = df[features]
y = df['user_preference']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred = model_multi.predict(X_test)

# Evaluasi
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2_multi = r2_score(y_test, y_pred)

st.subheader(" Koefisien Regresi Berganda")
coef_df = pd.DataFrame({
    'Fitur': features,
    'Koefisien': model_multi.coef_
}).sort_values(by='Koefisien', key=abs, ascending=False)
st.dataframe(coef_df, use_container_width=True)

st.subheader(" Evaluasi Model")
st.write(f"**Intercept:** {model_multi.intercept_:.3f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.3f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.3f}")
st.write(f"**R-squared (R²):** {r2_multi:.3f}")

# Plot: Prediksi vs Aktual
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred, ax=ax2)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("Prediksi vs Aktual")
st.pyplot(fig2)
