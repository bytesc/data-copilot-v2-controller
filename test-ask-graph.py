import re

from llm_access.LLM import get_llm
from config.get_config import config_data

import logging

import http.client
import json
import base64
from utils.write_csv import write_csv
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
        conn.request("POST", "/ask/graph", body=json_data, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()


if __name__ == "__main__":
    for _ in range(2):
        request = {
            "question": "查询员工姓名和工资",
            "concurrent": 1,
            "retries": 3
        }
        status_code, response = send_request(request)
        print(status_code, response)
        if status_code == 200:
            response = json.loads(response)
            filename = re.search(r"/(\w+\.png)", response['file'])[0]
            image_path = 'output_store/ask-graph'+filename
            with open(image_path, 'wb') as image_file:
                image_data = base64.b64decode(response['image_data'])
                image_file.write(image_data)
            print(f"Image saved to {image_path}")

            write_csv("output_store/data_log/ask_graph.csv",
                      [get_time(), request["question"], request["concurrent"], request["retries"], "/",
                       response['code'], response['retries_used'], response['file'], response["prompt"], "/",
                       "qwen1.5-110b-chat"])

        else:
            print("Failed to get image data.")




