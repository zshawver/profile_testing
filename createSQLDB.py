import sqlite3
import pandas as pd
import os

# File paths
results_csv_path  = "Output/2025-02-12_10-25_final_results.csv"  # Adjust as needed
summary_csv_path= "Output/2025-02-12_10-52_combinations_summary.csv"
db_path = "Output/results_db.sqlite"

# Remove the database if it already exists (optional)
if os.path.exists(db_path):
    os.remove(db_path)

# Connect to SQLite
conn = sqlite3.connect(db_path)

# Function to import CSV in chunks
def import_csv_to_sql(csv_path, table_name, conn, chunk_size=500000):
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk.to_sql(table_name, conn, if_exists="append", index=False)
        print(f"Inserted {len(chunk)} rows into {table_name}...")

# Import both files
import_csv_to_sql(results_csv_path, "full_results", conn)
import_csv_to_sql(summary_csv_path, "summary", conn, chunk_size=100000)  # Adjusted chunk size for smaller file

# Close connection
conn.close()
print("CSV successfully imported into SQLite 🎉")
