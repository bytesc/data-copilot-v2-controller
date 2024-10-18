import csv


def write_csv_from_list(file_name, data_list):
    with open(file_name, 'a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(data_list)
