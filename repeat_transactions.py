import pandas as pd

def repeat_rows_with_new_ids(input_file, output_file, repetitions, dynamic_col=""):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    init_max_customer = 7070
    final_max_customer = 42425
    
    if dynamic_col != "":
        max_id = df[dynamic_col].max()
    
    # Create a list to store the new rows
    new_rows = []
    current_customer = init_max_customer
    
    # Process each row
    for _, row in df.iterrows():
        for i in range(repetitions):
            new_row = row.copy()
            if repetitions > 0:
                max_id += 1
                new_row[dynamic_col] = max_id
                current_customer += 1
                if current_customer > final_max_customer:
                    current_customer = init_max_customer + 1
                new_row['senderID'] = current_customer

            new_rows.append(new_row)

    # Create a new DataFrame from the new rows
    new_df = pd.DataFrame(new_rows)
    
    # Write the new DataFrame to a CSV file
    new_df.to_csv(output_file, index=False)

# Call the function with desired file paths and repetitions
repeat_rows_with_new_ids('resources/data/financial_transactions.csv', 'resources/data/500_mb/financial_transactions.csv', 10, 'transactionID')

