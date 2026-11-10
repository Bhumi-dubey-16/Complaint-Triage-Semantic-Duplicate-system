import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

og=pd.read_csv("complaints.csv")

print(" the dataset null values :",og.isnull().sum())
og = og.dropna(subset=['Product', 'Issue'])
initial_count=len(og)
og=og.drop_duplicates()
final_count=len(og)

print("Product value counts:\n", og['Product'].value_counts())
print(f"removed no of duplicates:{initial_count}-{final_count}")


#train
train_ds,rest_ds=train_test_split(
  og,
  test_size=0.30,
  random_state=42,
  stratify=og['Product']
)

#predevl
val_ds, test_ds=train_test_split(
  rest_ds,
  test_size=0.50,
  random_state=42,
  stratify=rest_ds["Product"]
)
print("training set",train_ds.shape)
print("val ",val_ds.shape)
print("test dataset ",test_ds.shape)
print("saving all files...")
train_ds.to_csv("train_complaints.csv",index=False)
val_ds.to_csv("val_complaints.csv",index=False)
test_ds.to_csv("test_complaints.csv",index=False)
print("Files successfully saved!")