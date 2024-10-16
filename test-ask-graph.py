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
    for _ in range(3):
        request = {
            "question": "Which are the top 3 cities with the highest population in each country?",
            "concurrent": 1,
            "retries": 5
        }
        status_code2, response2 = prompt_request(request)
        status_code, response = send_request(request)
        print(status_code, status_code2)
        if status_code == 200 and status_code2 == 200:
            response = json.loads(response)
            response2 = json.loads(response2)
            if response['code']==200:
                filename = re.search(r"/(\w+\.png)", response['file'])[0]
                image_path = 'output_store/ask-graph'+filename
                with open(image_path, 'wb') as image_file:
                    image_data = base64.b64decode(response['image_data'])
                    image_file.write(image_data)
                print(f"Image saved to {image_path}")

            write_csv("output_store/data_log/ask_graph.csv",
                      [get_time(), request["question"], request["concurrent"], request["retries"], "/",
                       response['code'], response['retries_used'], response['file'],
                        response["success"], "/",
                       "qwen1.5-110b-chat", "/"])

            write_csv("output_store/ask-graph/" + response['file'] + ".txt",
                      [get_time(), request["question"], str(request), str(response), str(response2), "/",
                       "qwen1.5-110b-chat", "/"])
            print("Success")
        else:
            print("Failed to get image data.")




