import pandas as pd
import os

csv_file = "candidate_pairs_to_label.csv"
if not os.path.exists(csv_file):
    print(" candidate_pairs_to_label.csv not found!")
    exit()

df = pd.read_csv(csv_file)

if "Is_Duplicate" not in df.columns:
    df["Is_Duplicate"] = -1  # Initialize unlabled rows with -1

print("\n--- INTERACTIVE TERMINAL LABELER ---")
print("Instructions: Read the Query and Neighbor. Type '1' for Duplicate, '0' for Not Duplicate, or 'q' to quit and save.\n")

for idx, row in df.iterrows():
    if row["Is_Duplicate"] != -1:
        continue  # Skip already labeled rows if resuming
        
    print(f"\n[Pair {idx + 1} of {len(df)}] (Similarity: {row['Similarity']:.4f})")
    print(f"👉 Query:    {row['Query_Text']}")
    print(f"👉 Neighbor: {row['Neighbor_Text']}")
    
    while True:
        choice = input("Is this a duplicate? (1 = Yes, 0 = No, q = Quit): ").strip()
        if choice in ['1', '0']:
            df.loc[idx, "Is_Duplicate"] = int(choice)
            df.to_csv(csv_file, index=False)
            break
        elif choice.lower() == 'q':
            print("\n💾 Progress saved! Run the script again whenever you want to continue.")
            exit()
        else:
            print(" Invalid input. Please enter 1, 0, or q.")

print("\n All pairs labeled and saved successfully to 'candidate_pairs_to_label.csv'!")