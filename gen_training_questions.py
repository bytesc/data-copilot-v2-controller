import data_access.read_db
from llm_access import LLM, call_llm_test
from utils.write_csv import write_csv_from_list
import pandas as pd

pd.set_option('display.max_columns', None)


def fetch_data():
    dict_data, key, comments = data_access.read_db.get_data_from_db()
    return [dict_data, key, comments]


def slice_dfs(df_dict, lines=5):
    top_five_dict = {}
    for key, df in df_dict.items():
        top_five_dict[key] = df.head(min(lines, len(df)))
    return top_five_dict


def gen_questions(file_name, has_old=False, bach=5, lang="en"):
    data = fetch_data()
    if lang == "en":
        prompt = """I am trying to test a A natural language query system for intelligent,
         multi-table database queries, statistical computations, and chart generation.
         you’d better give questions that need multiple values output, such as give me top x or last y.
         try to give question that is suitable to use various types of graphs to illustrate.
         here is the test database structure:\n
        """ + str(slice_dfs(data[0])) + "Here is Key Constraints of the tables:" + str(data[1]) \
                 + f"please give me {bach} test case questions in english." + \
                 """
         the questions should be split by only a single \\n  
         you should only give me the output with out any additional explanations.
         here is an example:
         1. question text
         2. question text
         3. question text
        """
        if has_old:
            with open(file_name, "r", encoding="utf-8") as f:
                prompt = prompt + """
                These questions are already asked, do not ask same questions again:
                """ + str(f.readlines())

    else:
        prompt = ""

    llm = LLM.get_llm()
    ans = call_llm_test.call_llm(prompt, llm)

    try:
        ans_list = ans.split("\n")

        def remove_number_dot_space(s):
            import re
            return re.sub(r'^\d+\.\s+', '', s).replace("'", "").replace('"', '').strip()

        ans_list = [remove_number_dot_space(item) for item in ans_list]
        write_csv_from_list(file_name, ans_list)
        return True
    except Exception as e:
        print(e)
        return False


gen_questions("./gened_questions/training_questions_for_graph_1.csv", has_old=True, bach=10)
