from llm_access.LLM import get_llm

import logging


logging.basicConfig(filename='./ask_ai.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', encoding="utf-8")


llm = get_llm()


logging.info("setting up")


if __name__ == "__main__":
    logging.info("starting")
