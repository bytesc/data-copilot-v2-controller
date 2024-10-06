import csv


def write_csv(file_name, data_list):
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_list)
