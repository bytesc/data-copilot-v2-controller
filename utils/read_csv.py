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
