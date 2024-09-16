import pandas as pd

def repeat_rows_with_new_ids(input_file, output_file, repetitions, dynamic_col=""):
    dfTemp = pd.read_csv(input_file)
    max_id = dfTemp[dynamic_col].max()
    del dfTemp

    # Initial values for ID manipulation
    init_max_customer = 7070
    final_max_customer = 84851

    # Use chunksize to read the input CSV in smaller chunks
    chunk_size = 50000  # Adjust based on your memory constraints
    current_customer = init_max_customer

    # Open output file for writing
    with open(output_file, 'w', newline='') as f_out:
        # Iterate over the input file in chunks
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):

            # Create a list to store new rows for the current chunk
            new_rows = []
            for _, row in chunk.iterrows():
                for i in range(repetitions):
                    new_row = row.copy()
                    if i > 0:
                        max_id += 1
                        new_row[dynamic_col] = max_id
                    if i > 14:
                        current_customer += 1
                        if current_customer > final_max_customer:
                            current_customer = init_max_customer + 1
                        new_row['senderID'] = current_customer
                    new_rows.append(new_row)
            
            # Convert the new_rows list to a DataFrame
            new_df = pd.DataFrame(new_rows)
            
            # Write the DataFrame to CSV, appending after the first chunk
            new_df.to_csv(f_out, mode='a', header=f_out.tell() == 0, index=False)

# Call the function with desired file paths and repetitions
repeat_rows_with_new_ids('resources/data/financial_transactions.csv',
                         'resources/data/2_gb/financial_transactions.csv',
                         40, 'transactionID')

