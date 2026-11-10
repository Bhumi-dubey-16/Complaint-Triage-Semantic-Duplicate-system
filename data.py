import pandas as pd
def load_triage_data():
  """Loads the frozen, stratified train, validation, and test splits."""
  train_df=pd.read_csv("train_complaints.csv")
  val_df = pd.read_csv("val_complaints.csv")
  test_df = pd.read_csv("test_complaints.csv")
  return train_df, val_df, test_df


if __name__=="__main__":
  train,val,test=load_triage_data()
  print("Data loader verified!")
