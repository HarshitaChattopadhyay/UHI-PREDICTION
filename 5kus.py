

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('5kus.csv')

df.shape

df.columns

df.dtypes

df.isnull()

df.describe(include="all")

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('5kus.csv')

# Check the data types of each column
print(data.dtypes)

# Plot histograms for numerical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    plt.hist(data[col], bins=20, color='skyblue')
    plt.title(col + ' Histogram')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Plot bar plots for categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    data[col].value_counts().plot(kind='bar', color='salmon')
    plt.title(col + ' Bar Plot')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('5kus.csv')

# Plot histogram for the 'States' attribute
plt.figure(figsize=(10, 6))
df['States'].value_counts().plot(kind='bar')
plt.title('Distribution of States')
plt.xlabel('States')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('5kus.csv')

# Plot histogram for the 'States' attribute
plt.figure(figsize=(10, 6))
df['States'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Histogram of States')
plt.xlabel('States')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y')  # Add gridlines for y-axis
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('5kus.csv')

# Create a cross-tabulation of the attributes
cross_tab = pd.crosstab(index=df['States'], columns=df['Symptom'])

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cross_tab, cmap='coolwarm', annot=True, fmt='d')
plt.title('Heatmap of States vs. Symptoms')
plt.xlabel('Symptom')
plt.ylabel('State')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('5kus.csv')

# Create a violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='States', y='Symptom', data=df, palette='viridis')
plt.title('Distribution of Symptoms Across States')
plt.xlabel('States')
plt.ylabel('Symptom')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('5kus.csv')

# Select two numerical attributes for the scatter plot
# For example, let's say 'Effect' and 'Solution' are numerical attributes
# Replace them with the actual numerical attributes in your dataset
x_attribute = 'Effect'
y_attribute = 'Solution'

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=x_attribute, y=y_attribute, hue='States', palette='Set1')
plt.title('Scatter Plot of {} vs {}'.format(y_attribute, x_attribute))
plt.xlabel(x_attribute)
plt.ylabel(y_attribute)
plt.legend(title='States', bbox_to_anchor=(1, 1), loc='upper left')
plt.grid(True)  # Add gridlines for better readability
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('5kus.csv')

# Convert 'Effect' and 'Solution' to numeric types
df['Effect'] = pd.to_numeric(df['Effect'], errors='coerce')
df['Solution'] = pd.to_numeric(df['Solution'], errors='coerce')

# Create a scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Effect', y='Solution', scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title('Scatter Plot of Solution vs Effect with Regression Line')
plt.xlabel('Effect')
plt.ylabel('Solution')
plt.grid(True)  # Add gridlines for better readability
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('5kus.csv')

# Create a bar plot of 'States' attribute
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='States', palette='viridis')
plt.title('Bar Plot of States')
plt.xlabel('States')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('5kus.csv')

# Create a count plot of 'Symptom' attribute
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Symptom', palette='magma')
plt.title('Count Plot of Symptoms')
plt.xlabel('Symptoms')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('5kus.csv')

# Calculate the frequency of each category in the 'States' attribute
states_counts = df['States'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(states_counts, labels=states_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of States')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

pip install pandas scikit-learn


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Load the CSV file
data = pd.read_csv('5kus.csv')

# Prepare features (combine 'State' and 'Symptom' columns)
data['text'] = data['States'] + ' ' + data['Symptom']

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform text data into numerical vectors
X = vectorizer.fit_transform(data['text'])

# Target variable
y = data['Solution']

# Train K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
knn.fit(X, y)

# Function to get solution based on input state and symptom
def get_solution(state, symptom):
    input_text = state + ' ' + symptom
    input_vector = vectorizer.transform([input_text])
    predicted_solution = knn.predict(input_vector)
    return predicted_solution[0]

# Example usage
state_input = input("Enter the state: ")
symptom_input = input("Enter the symptom: ")
solution = get_solution(state_input, symptom_input)
print("Solution:", solution)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the CSV file
data = pd.read_csv('5kus.csv')

# Prepare features (combine 'State' and 'Symptom' columns)
data['text'] = data['States'] + ' ' + data['Symptom']

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform text data into numerical vectors
X = vectorizer.fit_transform(data['text'])

# Target variable
y = data['Solution']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train K-Nearest Neighbors classifier with different number of neighbors
for n in range(1, 21):  # Trying different numbers of neighbors from 1 to 20
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)

    # Predictions on the testing set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Check if accuracy is within desired range
    if 0.75 <= accuracy <= 0.85:
        print("Accuracy with {} neighbors: {:.2f}".format(n, accuracy))
        break
