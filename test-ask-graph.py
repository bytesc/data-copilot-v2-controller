import re
import time

from llm_access.LLM import get_llm
from config.get_config import config_data

import logging

import http.client
import json
import base64
from utils.write_csv import write_csv_from_list
from utils.read_csv import read_csv_to_list
from utils.get_time import get_time

logging.basicConfig(filename='./ask_ai.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', encoding="utf-8")

llm = get_llm()

logging.info("setting up")

conn = http.client.HTTPConnection(f"{config_data['server']['host']}:{config_data['server']['port']}")


def send_request(request_data):
    json_data = json.dumps(request_data)
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        conn.request("POST", "/ask/graph-2", body=json_data, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()


def prompt_request(request_data):
    json_data = json.dumps(request_data)
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        conn.request("POST", "/prompt/graph", body=json_data, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()


if __name__ == "__main__":
    question_list = read_csv_to_list("./gened_questions/training_questions_for_graph_1.csv")
    for question in question_list[:]:
        time.sleep(1)
        for _ in range(3):
            time.sleep(1)
            request = {
                "question": question,
                "concurrent": [1, 1],
                "retries": [5, 5]
            }
            status_code, response = send_request(request)
            print(status_code)
            if status_code == 200:
                response = json.loads(response)
                if response['code'] == 200:
                    filename = re.search(r"/(\w+\.png)", response['file'])[0]
                    image_path = 'output_store/ask-graph' + filename
                    with open(image_path, 'wb') as image_file:
                        image_data = base64.b64decode(response['image_data'])
                        image_file.write(image_data)
                    print(f"Image saved to {image_path}")

                write_csv_from_list("output_store/data_log/ask_graph_1.csv",
                                    [get_time(), request["question"],
                                     request["retries"][0], request["retries"][1], "/",
                                     response['code'],
                                     response['retries_used'][0], response['retries_used'][1],
                                     response['file'], "qwen1.5-110b-chat", "/"])
                if response['file']:
                    write_csv_from_list("output_store/ask-graph/" + response['file'] + ".txt",
                                        [get_time(), request["question"], str(request), str(response), "/",
                                         "qwen1.5-110b-chat", "/"])
                print("Success")
            else:
                print("Failed to get image data.")
