import math
import re
import string
import time
import random

from llm_access.LLM import get_llm
from config.get_config import config_data

from training.predict import predict

import concurrent.futures

import logging

import http.client
import json
import base64


def send_request(request_data):
    json_data = json.dumps(request_data)
    headers = {'Content-Type': 'application/json'}
    conn = None
    try:
        # 每个请求创建新连接（避免共享连接被关闭）
        conn = http.client.HTTPConnection(
            f"{config_data['server']['host']}:{config_data['server']['port']}"
        )
        conn.request("POST", "/ask/graph-2", body=json_data, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    except Exception as e:
        return None, str(e)
    finally:
        if conn:
            conn.close()  # 关闭当前请求的连接


def parse_and_save_image(response):
    status_code, json_data = response
    data = json.loads(json_data.decode('utf-8'))
    if status_code == 200 and data.get('code') == 200:
        image_base64 = data.get('image_data', '')
        if not image_base64:
            print("没有找到图像数据")
            return False

        try:
            # 解码Base64图像数据
            image_bytes = base64.b64decode(image_base64)

            def generate_random_string(length=8):
                letters = string.ascii_lowercase
                random_string = ''.join(random.choice(letters) for _ in range(length))
                return random_string
            # 保存图像文件
            image_path = './tmp_img/'+generate_random_string()+".png"
            with open(image_path, 'wb') as f:
                f.write(image_bytes)

            # print(f"图像已成功保存到 {image_path}")
            return image_path

        except Exception as e:
            print(f"保存图像时出错: {e}")
            return False
    else:
        print("响应状态码不正确或没有图像数据")
        return False


def calculate_optimal_threads(success_prob):
    if success_prob >= 0.999:  # 避免除以零或对数域错误
        return 1
    if success_prob <= 0:
        return 5  # 最低成功率 → 最大并发

    # # 计算满足 1 - (1 - p)^n ≥ 0.9 的最小n
    # required_threads = math.ceil(math.log(0.1) / math.log(1 - success_prob))
    # 计算满足 1 - (1 - p)^n ≥ 0.8 的最小n
    required_threads = math.ceil(math.log(0.2) / math.log(1 - success_prob))

    # 限制在 [1, 5] 范围内
    return max(1, min(required_threads, 5))


# def main():
#     question = "Display the distribution of the city populations within countries having more than 2 official languages using a scatter plot."
#     num = float(predict(question))
#     print(num)  # 0.48311216
#
#     num_threads = calculate_optimal_threads(num)
#     print(num_threads)
#
#     request = {
#         "question": question,
#         "concurrent": [1, 1],  # 这里可以保留原来的参数，或者调整
#         "retries": [1, 1]
#     }
#
#     # 使用 ThreadPoolExecutor 并发发送请求
#     results = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#         # 提交多个任务
#         futures = [executor.submit(send_request, request) for _ in range(num_threads)]
#
#         # 等待所有任务完成并收集结果
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 res = future.result()
#                 # print(res)
#                 results.append(res)
#             except Exception as e:
#                 print(f"Request failed: {e}")
#
#     # 处理所有结果（例如，只取第一个成功的结果，或合并结果）
#     if results:
#         i = 0
#         for result in results:
#             i += 1
#             status_code, json_data = result
#             data = json.loads(json_data.decode('utf-8'))
#             if status_code == 200 and data.get('code') == 200:
#                 image_path = parse_and_save_image(result)  # 这里可以优化，比如选择最佳结果
#                 print(i, image_path)
#             else:
#                 print(i, "执行失败")
#     else:
#         print("All requests failed.")

import concurrent.futures
import json
from pywebio.input import *
from pywebio.output import *
from pywebio.session import *
def main():
    put_markdown(f"# Data Copilot")
    while 1:
        # Get user input
        question = input("Enter your question", type=TEXT, required=True,
                         placeholder="")
        put_markdown("## " + question)
        # Display processing message
        put_text("Processing your question...")

        # Your existing processing code
        num = float(predict(question))
        put_markdown(f"**Prediction value:**")
        put_markdown(f"## {num}")

        num_threads = calculate_optimal_threads(num)
        put_markdown(f"**Using threads:**")
        put_markdown(f"## {num_threads}")

        request = {
            "question": question,
            "concurrent": [1, 1],
            "retries": [1, 1]
        }

        # Process with threads
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(send_request, request) for _ in range(num_threads)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    put_error(f"Request failed: {e}")

        # Display results
        if results:
            for i, result in enumerate(results, 1):
                status_code, json_data = result
                data = json.loads(json_data.decode('utf-8'))
                if status_code == 200 and data.get('code') == 200:
                    image_path = parse_and_save_image(result)
                    put_image(open(image_path, 'rb').read())  # Display the image
                    put_text(f"Result {i}: Success")
                else:
                    put_text(f"Result {i}: Failed")
        else:
            put_error("All requests failed.")

if __name__ == "__main__":
    # main()
    from pywebio import start_server

    start_server(main, port=8080)

# What are the top 6 cities with the highest population across all continents?
# List the countries with a surface area less than 100,000 sq km and their official languages.
# Display the average life expectancy in countries with a surface area larger than 500,000 square kilometers using a column chart.


# Provide the bottom 5 cities in terms of population from Europe.
# For countries with at least 5 million inhabitants, list the top 2 districts with the highest population.

# Chart the distribution of the population of cities in countries with more than 2 official languages using a histogram.
# Display the distribution of the city populations within countries having more than 2 official languages using a scatter plot.
