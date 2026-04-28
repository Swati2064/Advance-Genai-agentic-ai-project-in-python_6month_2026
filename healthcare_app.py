import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# load data
df = pd.read_csv(r"C:\Users\suraj\Downloads\heart.csv")   


sns.set_theme(style="whitegrid")
heart = sns.load_dataset("tips")

st.title('Miss Swati Jadhav seaborn bootcamp healthcare data visualization app')
st.write("This is a simple app to visualize the heathcare dataset using seaborn.")

# function create and display plot 
def display_plot(title, plot_func):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_func(ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    
# Title
st.title("Count Plot Example")

# Create figure
f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="target", data=df, ax=ax)

# Show in Streamlit
st.pyplot(f)

st.title("Count Plot: Sex vs Target")

# Create plot
f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="sex", hue="target", data=df, ax=ax)


# Show in Streamlit
st.pyplot(f)  

# Load data
# df = pd.read_csv("your_file.csv")

st.title("Catplot: Target vs Sex")

# Create catplot
g = sns.catplot(
    x="target",
    col="sex",
    data=df,
    kind="count",
    height=5,
    aspect=1
)

# Display in Streamlit
st.pyplot(g.fig)

st.title("Count Plot: Target vs Sex")

# Create plot
f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(y="target", hue="sex", data=df, ax=ax)

# Show in Streamlit
st.pyplot(f)

st.title("Count Plot (Target)")

# Create figure
f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="target", data=df, palette="Set3", ax=ax)

# Display in Streamlit
st.pyplot(f)

st.title("Styled Count Plot")

f, ax = plt.subplots(figsize=(8, 6))

sns.countplot(
    x="target",
    data=df,
    facecolor=(0, 0, 0, 0),  # transparent bars
    linewidth=3,
    edgecolor="black",       # single color (fix)
    ax=ax
)

st.pyplot(f)

st.title("Count Plot: Target vs FBS")

# Create plot
f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="target", hue="fbs", data=df, ax=ax)

# Display in Streamlit
st.pyplot(f)
 
st.title("Count Plot: Target vs Exang")

# Create plot
f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="target", hue="exang", data=df, ax=ax)

# Display in Streamlit
st.pyplot(f)

f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="cp", hue="target", data=df)
plt.show()

f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="cp", data=df)
plt.show()

st.title("Count Plot: Chest Pain (cp) vs Target")

# Create plot
f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="cp", hue="target", data=df, ax=ax)

# Labels (optional but recommended)
ax.set_title("Chest Pain Type vs Target")
ax.set_xlabel("Chest Pain Type (cp)")
ax.set_ylabel("Count")
ax.legend(title="Target")

# Show in Streamlit
st.pyplot(f)

st.title("Count Plot: Chest Pain (cp)")

f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="cp", data=df, ax=ax)

st.pyplot(f)

st.title("Catplot: Target vs Chest Pain (cp)")

# Create catplot
g = sns.catplot(
    x="target",
    col="cp",
    data=df,
    kind="count",
    height=8,
    aspect=1
)

# Display in Streamlit
st.pyplot(g.fig)

# df = pd.read_csv("your_file.csv")

st.title("Distribution Plot: Thalach")

# Create figure
f, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    df['thalach'],
    bins=10,
    kde=True,   # adds smooth curve
    ax=ax
)

ax.set_title("Distribution of Thalach")
ax.set_xlabel("Thalach")
ax.set_ylabel("Frequency")

# Show in Streamlit
st.pyplot(f)

st.title("Distribution Plot: Thalach")

# Prepare data
x = df['thalach']
x = pd.Series(x, name="thalach variable")

# Create plot
f, ax = plt.subplots(figsize=(10, 6))
sns.histplot(x, bins=10, kde=True, ax=ax)

ax.set_title("Distribution of Thalach")
ax.set_xlabel("Thalach")
ax.set_ylabel("Frequency")

# Display in Streamlit
st.pyplot(f)

st.title("Vertical Distribution Plot: Thalach")

x = df['thalach']

# Create figure
f, ax = plt.subplots(figsize=(10, 6))

# Use y= for vertical histogram
sns.histplot(
    y=x,
    bins=10,
    kde=True,
    ax=ax
)

ax.set_title("Vertical Distribution of Thalach")
ax.set_xlabel("Frequency")
ax.set_ylabel("Thalach")

# Display in Streamlit
st.pyplot(f)

st.title("KDE Plot: Thalach")

# Prepare data
x = df['thalach']
x = pd.Series(x, name="thalach variable")

# Create plot
f, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(x=x, ax=ax)

ax.set_title("Density Plot of Thalach")
ax.set_xlabel("Thalach")
ax.set_ylabel("Density")

# Display in Streamlit
st.pyplot(f)

st.title("KDE Plot: Thalach")

# Prepare data
x = df['thalach']
x = pd.Series(x, name="thalach variable")

# Create plot
f, ax = plt.subplots(figsize=(10, 6))

sns.kdeplot(
    x=x,
    fill=True,      # replaces shade=True
    color='r',
    ax=ax
)

ax.set_title("Density Plot of Thalach")
ax.set_xlabel("Thalach")
ax.set_ylabel("Density")

# Display in Streamlit
st.pyplot(f)

st.title("Histogram + Rug Plot: Thalach")

x = df['thalach']

# Create figure
f, ax = plt.subplots(figsize=(10, 6))

# Histogram (no KDE)
sns.histplot(x, bins=10, ax=ax)

# Rug plot (small ticks)
sns.rugplot(x, ax=ax)

ax.set_title("Thalach Distribution")
ax.set_xlabel("Thalach")
ax.set_ylabel("Frequency")

# Display in Streamlit
st.pyplot(f)

st.title("Strip Plot")

f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=df, ax=ax)

st.pyplot(f)

st.title("Strip Plot (with jitter)")

f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=df, jitter=0.01, ax=ax)

st.pyplot(f)

st.title("Box Plot: Target vs Thalach")

f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="target", y="thalach", data=df, ax=ax)

st.pyplot(f)

st.title("Correlation Heatmap of Heart Disease Dataset")

# Compute correlation (only numeric columns)
correlation = df.select_dtypes(include='number').corr()

# Create figure
f, ax = plt.subplots(figsize=(16, 12))

sns.heatmap(
    correlation,
    square=True,
    annot=True,
    fmt='.2f',
    linecolor='white',
    cmap='coolwarm',
    ax=ax
)

# Rotate labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)

# Display in Streamlit
st.pyplot(f)

st.title("Pair Plot")

num_var = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']

# Create pairplot
g = sns.pairplot(df[num_var], kind='scatter', diag_kind='hist')

# Display in Streamlit
st.pyplot(g.fig)

st.title("Age Distribution")

f, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    df['age'],
    bins=10,
    kde=True,
    ax=ax
)

ax.set_title("Distribution of Age")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")

st.pyplot(f)

st.title("Strip Plot: Target vs Age")

f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="age", data=df, ax=ax)

st.pyplot(f)

st.title("Box Plot: Target vs Age")

f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="target", y="age", data=df, ax=ax)

st.pyplot(f)

st.title("Scatter Plot: Age vs Trestbps")

f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="age", y="trestbps", data=df, ax=ax)

st.pyplot(f)

st.title("Regression Plot: Age vs Trestbps")

f, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x="age", y="trestbps", data=df, ax=ax)

st.pyplot(f)

st.title("Scatter Plot: Age vs Cholesterol")

f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="age", y="chol", data=df, ax=ax)

st.pyplot(f)

st.title("Regression Plot: Age vs Cholesterol")

f, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x="age", y="chol", data=df, ax=ax)

st.pyplot(f)

st.title("Scatter Plot: Cholesterol vs Thalach")

f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="chol", y="thalach", data=df, ax=ax)

st.pyplot(f)

st.title("Regression Plot: Cholesterol vs Thalach")

f, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x="chol", y="thalach", data=df, ax=ax)

st.pyplot(f)

st.title("Box Plot: Age")

f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["age"], ax=ax)

st.pyplot(f)

st.title("Box Plot: Trestbps")

f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["trestbps"], ax=ax)

st.pyplot(f)

st.title("Box Plot: Cholesterol")

f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["chol"], ax=ax)

st.pyplot(f)

st.title("Box Plot: Thalach")

f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["thalach"], ax=ax)

st.pyplot(f)

st.title("Box Plot: Oldpeak")

f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["oldpeak"], ax=ax)

st.pyplot(f)