



import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('weather.csv')

#DATA EXPLORATION

#checking the datas
df.shape

df.columns

df.dtypes

### Looking for missing values

df.isnull()

df.isnull().any()

df.isnull().sum()

### Putting it all together

df.info()

### Looking for duplicates

df.duplicated()

df.duplicated().sum()

### Statistical description of the data

df.describe()

df.describe(include="all")

# Exploring the data using Visualizations

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
weather_data = pd.read_csv('weather.csv')

# Select relevant attributes
relevant_attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC', 'precipitation', 'Wind Speed (km/h)']

# Plot histograms for each attribute
for attribute in relevant_attributes:
    plt.figure(figsize=(8, 6))
    plt.hist(weather_data[attribute], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {attribute}', fontsize=16)
    plt.xlabel(attribute, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
weather_data = pd.read_csv("weather.csv")

# Plot a histogram for precipitation
plt.figure(figsize=(10, 6))
sns.histplot(data=weather_data, x='precipitation', bins=20, kde=True, color='skyblue')
plt.title('Distribution of Precipitation')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("weather.csv")

# Select relevant attributes
relevant_attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']

# Plot boxplot
plt.figure(figsize=(10, 6))
df[relevant_attributes].boxplot()
plt.title('Temperature Changes by Month')
plt.ylabel('Temperature (°C)')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('weather.csv')

# Select relevant attributes
relevant_attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC', 'precipitation',
                       'TemperaturePerYr', 'Apparent Temperature (C)', 'Wind Speed (km/h)']

# Create a boxen plot for selected attributes
plt.figure(figsize=(12, 8))
sns.boxenplot(data=df[relevant_attributes])
plt.title('Boxen Plot of Relevant Attributes')
plt.xlabel('Attributes')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('weather.csv')

# Select relevant attributes
relevant_attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC', 'precipitation',
                       'TemperaturePerYr', 'Apparent Temperature (C)', 'TempMax', 'tempMin',
                       'wind', 'Wind Speed (km/h)', 'Visibility (km)']

# Create violin plot for 'JAN-FEB'
plt.figure(figsize=(10, 6))
sns.violinplot(x='JAN-FEB', y='TemperaturePerYr', data=df)
plt.title('Temperature Changes Distribution for JAN-FEB')
plt.xlabel('JAN-FEB')
plt.ylabel('Temperature Changes (°C)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('weather.csv')

# Select relevant attributes
relevant_attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC', 'precipitation',
                       'TemperaturePerYr', 'Apparent Temperature (C)', 'TempMax', 'tempMin',
                       'wind', 'Wind Speed (km/h)', 'Visibility (km)']

# Create violin plot for 'MAR-MAY'
plt.figure(figsize=(10, 6))
sns.violinplot(x='MAR-MAY', y='TemperaturePerYr', data=df)
plt.title('Temperature Changes Distribution for MAR-MAY')
plt.xlabel('MAR-MAY')
plt.ylabel('Temperature Changes (°C)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('weather.csv')

# Select relevant attributes
relevant_attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC', 'precipitation',
                       'TemperaturePerYr', 'Apparent Temperature (C)', 'TempMax', 'tempMin',
                       'wind', 'Wind Speed (km/h)', 'Visibility (km)']

# Create violin plot for 'OCT-DEC'
plt.figure(figsize=(10, 6))
sns.violinplot(x='OCT-DEC', y='TemperaturePerYr', data=df)
plt.title('Temperature Changes Distribution for OCT-DEC')
plt.xlabel('OCT-DEC')
plt.ylabel('Temperature Changes (°C)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('weather.csv')

# Select relevant attributes
attributes = ['precipitation', 'wind', 'Visibility (km)', 'TemperaturePerYr']

# Filter out rows with missing values in selected attributes
data_filtered = data.dropna(subset=attributes)

# Create scatter plot
plt.figure(figsize=(10, 6))

# Loop through each attribute and plot against TemperaturePerYr
for attribute in attributes[:-1]:  # Exclude 'TemperaturePerYr'
    plt.scatter(data_filtered[attribute], data_filtered['TemperaturePerYr'], label=attribute)

# Add labels and title
plt.xlabel('Attribute Value')
plt.ylabel('Temperature Change')
plt.title('Scatter Plot of Temperature Change vs. Selected Attributes')
plt.legend()

# Show plot
plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('weather.csv')

# Select relevant attributes
attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC', 'precipitation', 'wind', 'Visibility (km)', 'TempMax', 'tempMin']

# Drop rows with missing values
data = data.dropna(subset=attributes)

# Split the data into training and testing sets
X = data[attributes]
y = data['TemperaturePerYr']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot actual vs. predicted temperature changes
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel('Actual Temperature Changes')
plt.ylabel('Predicted Temperature Changes')
plt.title('Linear Regression: Actual vs. Predicted Temperature Changes')
plt.grid(True)
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
weather_df = pd.read_csv("weather.csv")

# Select relevant attributes
relevant_attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC', 'precipitation',
                       'TemperaturePerYr', 'Apparent Temperature (C)', 'Wind Speed (km/h)',
                       'Visibility (km)']

# Create pairplot
sns.pairplot(weather_df[relevant_attributes])
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
weather_data = pd.read_csv("weather.csv")

# Select relevant attributes
relevant_attributes = ["JAN-FEB", "MAR-MAY", "JUN-SEP", "OCT-DEC", "precipitation", "TemperaturePerYr", "wind", "Visibility (km)"]

# Calculate correlation matrix
correlation_matrix = weather_data[relevant_attributes].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Relevant Attributes')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
weather = pd.read_csv("weather.csv")

# Choose a categorical attribute for visualization (e.g., PrepType)
# Replace 'PrepType' with the attribute you want to visualize
sns.countplot(data=weather, x='PrepType')
plt.title('Count of Different Precipitation Types')
plt.xlabel('Precipitation Type')
plt.ylabel('Count')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
weather_data = pd.read_csv("weather.csv")

# Select relevant attributes
relevant_attributes = ["JAN-FEB", "MAR-MAY", "JUN-SEP", "OCT-DEC", "precipitation", "TemperaturePerYr"]

# Subset the data with selected attributes
relevant_data = weather_data[relevant_attributes]

# Calculate average temperature changes for each attribute
average_temp_changes = relevant_data.mean()

# Plot the bar plot
plt.figure(figsize=(10, 6))
average_temp_changes.plot(kind='bar', color='skyblue')
plt.title('Average Temperature Changes by Attribute')
plt.xlabel('Attribute')
plt.ylabel('Average Temperature Change')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
weather_data = pd.read_csv("weather.csv")

# Select relevant attributes
selected_attributes = ["JAN-FEB", "MAR-MAY", "JUN-SEP", "OCT-DEC", "precipitation", "Wind Speed (km/h)"]

# Calculate the average values for each attribute
avg_values = weather_data[selected_attributes].mean()

# Plot a pie chart
plt.figure(figsize=(8, 8))
plt.pie(avg_values, labels=avg_values.index, autopct='%1.1f%%', startangle=140)
plt.title('Average Values of Selected Attributes')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
weather_df = pd.read_csv("weather.csv")

# Select relevant attributes
attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC', 'precipitation', 'Wind Speed (km/h)', 'weather']

# Create a subset of the dataframe with selected attributes
weather_subset = weather_df[attributes]

# Drop rows with missing values
weather_subset.dropna(inplace=True)

# FacetGrid with pairplot for visualization
g = sns.PairGrid(weather_subset, hue='weather')
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot, kde=True)
g.add_legend()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
weather = pd.read_csv('weather.csv')

# Select relevant attributes
relevant_attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC', 'precipitation', 'Wind Speed (km/h)', 'Visibility (km)']

# Plot a stripplot for each attribute
plt.figure(figsize=(12, 8))
for i, attribute in enumerate(relevant_attributes):
    plt.subplot(3, 3, i+1)
    sns.stripplot(x=weather[attribute], jitter=True)
    plt.title(attribute)
    plt.xlabel('')
    plt.ylabel('Temperature')
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
weather = pd.read_csv('weather.csv')

# Select relevant attributes
relevant_attributes = ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC', 'precipitation', 'TemperaturePerYr', 'wind', 'Visibility (km)']

# Filter the dataset to include only relevant attributes
relevant_data = weather[relevant_attributes]

# Plot the swarmplot
plt.figure(figsize=(12, 8))
sns.swarmplot(data=relevant_data)
plt.title('Swarmplot of Relevant Attributes')
plt.xlabel('Attributes')
plt.ylabel('Values')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()
