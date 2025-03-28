# EXNO2DS
## Register Number: 212223100025
## Name: G Lekasri
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/My Drive/titanic_dataset.csv"
```
![Screenshot 2025-03-27 155603](https://github.com/user-attachments/assets/fcd3b597-b566-45e4-b5d1-f109f81ef16c)
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(path)
df
```
![Screenshot 2025-03-27 155650](https://github.com/user-attachments/assets/06efa84e-f185-478d-9598-e727bbb095e4)
```
df.info()
```
![Screenshot 2025-03-27 155703](https://github.com/user-attachments/assets/56504935-83da-400b-9475-4d5ee5c22887)
```
df.shape
```
![Screenshot 2025-03-27 155710](https://github.com/user-attachments/assets/da9402a4-9c1c-4ae7-9bc9-aaa3f9e0ad50)
```
df.head(5)
```
![Screenshot 2025-03-27 155728](https://github.com/user-attachments/assets/da7cb8a2-b672-4286-a09e-78384dd82b8a)
```
df.describe()
```
![Screenshot 2025-03-27 155740](https://github.com/user-attachments/assets/08aa3770-731c-4dd5-b271-b553d0d53650)
```
df.set_index("PassengerId", inplace=True)
df.describe()
```
![Screenshot 2025-03-27 155749](https://github.com/user-attachments/assets/20ffd918-676f-4fe2-9fd5-278088162519)
```
df.shape
```
![Screenshot 2025-03-27 155758](https://github.com/user-attachments/assets/c83c8998-d8e8-4eed-a5aa-a7bf652ead10)
```
df.nunique()
```
![Screenshot 2025-03-27 155808](https://github.com/user-attachments/assets/55e8ca41-c256-4c49-9ef7-75337c38b4cb)
```
df["Survived"].value_counts()
```
![Screenshot 2025-03-27 155815](https://github.com/user-attachments/assets/d192e6e8-1793-4830-87e8-cd169c492e75)
```
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per

```
![Screenshot 2025-03-27 155825](https://github.com/user-attachments/assets/71a8491a-982d-4c7e-8153-6cbc6452c3b5)
```
sns.countplot(data=df,x="Survived")
```
![Screenshot 2025-03-27 155833](https://github.com/user-attachments/assets/0e958cab-f086-49e2-872d-ce2bf3f9dc6e)
```
df.Pclass.unique()
```
![Screenshot 2025-03-27 155847](https://github.com/user-attachments/assets/82ae014c-9c77-4462-95c0-d93142d8fbb4)
```
df.rename(columns={'Sex':'Gender'},inplace=True)
df
```
![Screenshot 2025-03-27 155903](https://github.com/user-attachments/assets/95d54f90-21ee-44ab-8e82-d2da17337f35)
```
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)
```
![Screenshot 2025-03-27 155919](https://github.com/user-attachments/assets/2123599f-2b9e-442b-93fa-d1a088086e62)
```
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
```
![Screenshot 2025-03-27 155933](https://github.com/user-attachments/assets/e25f2f15-80e5-4692-9c28-c118a33772db)
```
df.boxplot(column="Age",by="Survived")
```
![Screenshot 2025-03-27 155946](https://github.com/user-attachments/assets/9a83d527-2d31-4df6-8520-bf1784b85f30)
```
sns.scatterplot(x=df["Age"],y=df["Fare"])
```
![Screenshot 2025-03-27 155956](https://github.com/user-attachments/assets/bb3fb353-273f-4f87-a225-c03fef0178e5)
```
sns.jointplot(x="Age",y="Fare",data=df)
```
![Screenshot 2025-03-27 160012](https://github.com/user-attachments/assets/b82cd5cb-29d0-4ad1-9918-fcac08fef3c9)
```
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax1 = plt.subplots(figsize=(8,5)) 
sns.boxplot(ax=ax1, x='Pclass', y='Age', hue='Gender', data=df) 
plt.show()
```
![Screenshot 2025-03-27 160037](https://github.com/user-attachments/assets/08700ad5-9042-42ad-80d3-297da9ada7bd)
```
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![Screenshot 2025-03-27 160056](https://github.com/user-attachments/assets/542b594e-e416-4896-a146-a1cf6d6829ba)
```
numerical_features = df.select_dtypes(include=np.number).columns
corr = df[numerical_features].corr()
sns.heatmap(corr, annot=True)
sns.pairplot(df)
```
![Screenshot 2025-03-27 160151](https://github.com/user-attachments/assets/008bc2a5-1a85-42bb-abf6-8156c9d0751c)

# RESULT
      The code has been run and executed successfully
