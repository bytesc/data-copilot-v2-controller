from utils.read_csv import read_csv_to_list_row


def process_output_list(output_file):
    output_list = read_csv_to_list_row(output_file)
    output_list = [[row[1], row[5], row[6], row[7], row[8]] for row in output_list]
    # print(output_list)
    unique_questions = list(set([row[0] for row in output_list]))
    # print(unique_questions)
    outcome = {}
    for question in unique_questions:
        right = 0
        wrong = 0
        retry_list = [0 for _ in range(6)]
        outputs = [row for row in output_list if row[0] == question]
        file = ""
        for result in outputs:
            if result[1] == "200" or (result[1] == "504" and result[3] != ""):
                right = right+1
                retry_list[int(result[2])] = retry_list[int(result[2])]+1
                file = result[4]
            else:
                wrong = wrong + 1
        outcome[question] = [retry_list, right/(wrong+right), file]
    print(outcome)
    return outcome


if __name__ == "__main__":
    output_file_name = "../output_store/data_log/ask_graph_1.csv"
    process_output_list(output_file_name)
