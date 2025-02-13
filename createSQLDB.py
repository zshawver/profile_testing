import sqlite3
import pandas as pd
import os

# File paths
results_csv_path  = "Output/2025-02-12_10-25_final_results.csv"
summary_csv_path = "Output/2025-02-12_10-52_combinations_summary.csv"
db_path = "Output/results_db.sqlite"

# Remove the database if it already exists
if os.path.exists(db_path):
    os.remove(db_path)

# Connect to SQLite
conn = sqlite3.connect(db_path)

# Function to import CSV in chunks (Ensures Schema Consistency)
def import_csv_to_sql(csv_path, table_name, conn, chunk_size=500000):
    first_chunk = True  # Track if it's the first chunk
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Ensure that if_exists="replace" is only used for the first chunk
        chunk.to_sql(table_name, conn, if_exists="replace" if first_chunk else "append", index=False)
        first_chunk = False  # Set to False after first write
        print(f"Inserted {len(chunk)} rows into {table_name}...")

# Import full results
import_csv_to_sql(results_csv_path, "full_results", conn)

# Import summary results
import_csv_to_sql(summary_csv_path, "summary", conn, chunk_size=100000)

# Check SQLite table structure after import
with conn:
    cursor = conn.execute("PRAGMA table_info(full_results);")
    # columns = [row[1] for row in cursor.fetchall()]
    # print("Columns in SQLite full_results:", columns)

    cursor = conn.execute("PRAGMA table_info(summary);")
    summary_columns = [row[1] for row in cursor.fetchall()]
    print("Columns in SQLite summary:", summary_columns)

# Close connection
conn.close()
print("CSV successfully imported into SQLite 🎉")
