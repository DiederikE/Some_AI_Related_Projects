# Read the content of the first file
with open("pair0001.txt", "r") as file1:
    lines1 = file1.readlines()

# Read the content of the second file
with open("pair0002.txt", "r") as file2:
    lines2 = file2.readlines()

# Create a list to store the merged data
merged_data = []

# Iterate over the lines in both files
for line1, line2 in zip(lines1, lines2):
    # Split the lines by space
    data1 = line1.strip().split()
    data2 = line2.strip().split()

    # Merge the data from both files
    merged_line = f"{data1[0]} {data1[1]} {data2[1]}\n"
    merged_data.append(merged_line)

# Write the merged data to a new file
with open("alt_temp_preci.txt", "w") as merged_file:
    merged_file.writelines(merged_data)

# Print a success message
print("The data from file1.txt and file2.txt has been merged into alt_temp_preci.txt with 3 columns.")
