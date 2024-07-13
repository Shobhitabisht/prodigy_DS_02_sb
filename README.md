Performing data cleaning and exploratory data analysis (EDA) on the Titanic dataset involves several steps to understand the dataset, handle missing values, explore relationships between variables, and identify patterns and trends. Here's a structured approach to do this:

Step 1: Load the Dataset
First, load the dataset and necessary libraries for analysis.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('train.csv')

Step 2: Explore the Data
Understanding the structure of the dataset:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('train.csv')

Check for missing values:
df.isnull().sum()  # Count missing values in each column

Step 3: Data Cleaning
Handle missing values:
# Cabin has many missing values, drop it for simplicity
df = df.drop(columns=['Cabin'])

# Fill missing values in Age with median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Fill missing values in Embarked with the mode (most frequent value)
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

Convert categorical variables to numeric:

python
Copy code
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
Step 4: Exploratory Data Analysis (EDA)
Univariate Analysis:

python
Copy code
# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.show()

# Count of survivors
sns.countplot(x='Survived', data=df)
plt.title('Count of Survivors')
plt.show()
Bivariate Analysis:

python
Copy code
# Survival rate by sex
sns.barplot(x='Sex', y='Survived', data=df, ci=None)
plt.title('Survival Rate by Sex')
plt.show()

# Survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=df, ci=None)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Survival rate by age
sns.histplot(x='Age', hue='Survived', data=df, kde=True)
plt.title('Survival Rate by Age')
plt.show()
Correlation Matrix:

python
Copy code
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()




thankyou
{shobhita bisht}
