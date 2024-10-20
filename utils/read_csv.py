import csv


def read_csv_to_list(file_path):
    cell_list = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for cell in row:
                cell_list.append(cell)
    return cell_list

# cell_list = read_csv_to_list('data.csv')
# print(cell_list)


def read_csv_to_list_row(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    return data

# Example usage (this won't work here as we don't have access to files in this environment)
# data = read_csv_to_list("path_to_your_file.csv")
# print(data)
