import requests
import json

from weaver.utils.config import config
from weaver.utils.html_parse import getWebContent


def execute_web_query(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
      "q": query
    })
    headers = {
      'X-API-KEY': config["api_keys"]["X_API_KEY"],
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)


    def extract_top_links(data, top_n=5):
        links = []

        # 提取 organic 结果中的 snippet
        if "organic" in data:
            for item in data["organic"]:  # 取前top_n条
                if top_n>0:
                    if "snippet" in item and "pdf" not in item["link"] and "abstract" not in  item["link"] and "articles" not in item["link"]:
                        links.append(item["link"])
                        top_n -= 1
        # 拼接文本
        return links


    # 示例 JSON 字符串
    data_str = response.text

    data = json.loads(data_str)
    links = extract_top_links(data)
    texts_trunck=[]
    for url in links:
      print(url)
      try:
        for text in getWebContent(url):
          if "'Just a moment" not in text and "To ensure we keep this website safe" not in text:
            texts_trunck.append(text)
      except Exception as e:
        print(e)

    return texts_trunck

if __name__ == "__main__":
    query = "Galactic thin disk"
    result_text = execute_web_query(query)
    print(len(result_text))
