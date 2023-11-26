import csv
import random
import sys

def generate_random_values(n):
    # Generate n rows with random values between 0 and 360 for each column
    data = [(random.uniform(0, 360), random.uniform(0, 360)) for _ in range(n)]
    return data

def save_to_csv(filename, data):
    # Save data to a CSV file with two columns
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Column1', 'Column2'])  # Write header
        csv_writer.writerows(data)

if __name__ == "__main__":
    # Check if the number of command-line arguments is correct
    if len(sys.argv) != 2:
        sys.exit(1)

    # Get the number of rows from the command-line argument
    n_rows = int(sys.argv[1])

    # Generate random values and save to CSV
    random_data = generate_random_values(n_rows)
    save_to_csv('data.csv', random_data)
