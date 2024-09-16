import pandas as pd

def repeat_rows_with_new_ids(input_file, output_file, repetitions, dynamic_col=""):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    if dynamic_col != "":
        max_id = df[dynamic_col].max()
    print("Initial max customer id: ", max_id)
    
    # Create a list to store the new rows
    new_rows = []
    
    # Process each row
    for _, row in df.iterrows():
        for i in range(repetitions):
            new_row = row.copy()
            if i > 0:
                max_id += 1
                new_row[dynamic_col] = max_id
            new_rows.append(new_row)

    print("Final max customer id: ", max_id)
    # Create a new DataFrame from the new rows
    new_df = pd.DataFrame(new_rows)
    
    # Write the new DataFrame to a CSV file
    new_df.to_csv(output_file, index=False)

# Call the function with desired file paths and repetitions
repeat_rows_with_new_ids('resources/data/customer.csv', 'resources/data/1_gb/customer.csv', 7, 'c_customer_sk')


