import requests
import json

from weaver.utils.config import config
from weaver.utils.html_parse import getWebContent


def execute_web_query(query):
    url = "https://api.bochaai.com/v1/web-search"

    payload = json.dumps({
      "query": query,
      "summary": True,
      "count": 5
    })
    headers = {
      'Authorization': f'Bearer {config["api_keys"]["X_API_KEY"]}',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    def extract_top_links(response_data, top_n=5):
        links = []
        
        # 检查响应格式
        if response_data.get("code") != 200 or "data" not in response_data:
            print(f"搜索API返回错误: {response_data.get('msg', '未知错误')}")
            return links
            
        search_data = response_data["data"]
        
        # 从新的API响应格式中提取链接
        web_pages = search_data.get("webPages", {}).get("value", [])
        
        # 提取链接
        for page in web_pages:
            if top_n > 0:
                url = page.get("url")
                excluded_keywords = ("pdf", "abstract", "articles", "image", "view", "video")
                if url and not any(keyword in url.lower() for keyword in excluded_keywords):
                    links.append(url)
                    top_n -= 1
                    
        return links

    # 解析响应
    data = response.json()
    links = extract_top_links(data)
    texts_trunck=[]
    for url in links:
      print(url)
      try:
        for text in getWebContent(url):
          if "'Just a moment" not in text and "To ensure we keep this website safe" not in text and "COOKIES" not in text:
            texts_trunck.append(text)
      except Exception as e:
        print(e)

    return texts_trunck

if __name__ == "__main__":
    query = "Galactic thin disk"
    result_text = execute_web_query(query)
    print(len(result_text))
