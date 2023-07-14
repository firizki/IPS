import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("datasets/merged_dataset/label.csv")

unq_values = df["name"].unique()
print("Total Records: ", len(df))
print("Unique Images: ",len(unq_values))

null_values = df.isnull().sum(axis = 0)
print("\n> Null Values in each column <")
print(null_values)

classes = df["classname"].unique()
print("Total Classes: ",len(classes))
print("\n> Classes <\n",classes)

plt.figure(figsize=(14,8))
plt.title('Class Distribution', fontsize= 20)
sns.countplot(x = "classname", data = df)
# plt.show()