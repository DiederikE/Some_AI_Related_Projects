# Open all the files in read mode
with open("pair0001.txt", "r") as file1, open("pair0002.txt", "r") as file2, open("pair0003.txt", "r") as file3, open("pair0004.txt", "r") as file4:
    # Read the lines from each file
    lines1 = file1.readlines()
    lines2 = file2.readlines()
    lines3 = file3.readlines()
    lines4 = file4.readlines()

# Create a list to store the merged data
merged_data = []

# Iterate over the lines in all files
for line1, line2, line3, line4 in zip(lines1, lines2, lines3, lines4):
    # Split the lines by space
    data1 = line1.strip().split()
    data2 = line2.strip().split()
    data3 = line3.strip().split()
    data4 = line4.strip().split()

    # Merge the data from all files
    merged_line = f"{data1[0]} {data1[1]} {data4[1]}\n"
    merged_data.append(merged_line)

# Write the merged data to a new file
with open("alt_temp_sun.txt", "w") as merged_file:
    merged_file.writelines(merged_data)

print("The data from pair0001.txt, pair0002.txt, pair0003.txt and pair0004.txt has been merged into alt_temp_prec_long_sun.txt.")