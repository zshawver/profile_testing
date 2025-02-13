import sqlite3
import pandas as pd
import ast
import matplotlib.pyplot as plt

db_path = "Output/results_db.sqlite"  # Update this if needed

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()

    # Get the max num_filtered_results
    cursor.execute("SELECT MAX(num_filtered_results) FROM summary;")
    max_value = cursor.fetchone()[0]
    print("Max num_filtered_results:", max_value)

    # Get rows from full_results matching max value
    query = """
        SELECT * FROM full_results
        WHERE num_filtered_results = ?;
    """
    df = pd.read_sql_query(query, conn, params=(max_value,))

# Display results as a DataFrame
print(df)

df.columns

for juror in df['matched_juror']:
    print(type(juror))


df.to_excel("Output/MatchedJurors_MostResults.xlsx", index = False)

#Get the number of rows in the full_results table
conn = sqlite3.connect(db_path)  # Replace with your actual database file
cursor = conn.cursor()

# Execute the count query
cursor.execute("SELECT COUNT(*) FROM full_results")
row_count = cursor.fetchone()[0]

print("Number of rows in full_results:", row_count)

# Close the connection when done
conn.close()


# df["matched_juror"] = df["matched_juror"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

conn = sqlite3.connect(db_path)  # Replace with your actual database file
cursor = conn.cursor()

# Query the summary table
cursor.execute("SELECT num_jurors_matched, num_filtered_results FROM summary")  # Replace with your actual table name
data = cursor.fetchall()
conn.close()


# Extract data into separate lists
num_jurors_matched, num_filtered_results = zip(*data)  # Unzips into two lists

# Create scatter plot
plt.figure(figsize=(16, 12))
plt.scatter(num_jurors_matched, num_filtered_results,
            color="darkgrey", edgecolors="yellow",
            alpha=0.7, linewidth=1.2)
# Labels and title with increased font size
plt.xlabel("Number of Jurors Matched", fontsize=24)
plt.ylabel("Number of Filtered Results", fontsize=24)
# plt.title("Juror Matches vs. Filtered Results", fontsize=16)

# Increase tick label font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

# Show the plot
plt.show()
