import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter

from weaver.utils.config import config


def getWebContent(url):
    # 定义 headers，模拟浏览器访问
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
    }
    response = requests.get(url,headers= headers,timeout=60)
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取页面中的所有文本
    cleaned_text = soup.get_text(separator=" ", strip=True)

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = config['chunk']['chunk_size'],
        chunk_overlap  = config['chunk']['chunk_overlap'],
        length_function = len,
    )
    texts = text_splitter.create_documents([cleaned_text])
    truncked_texts = []
    for item in texts:
        truncked_texts.append(item.page_content)
    return truncked_texts

if __name__ == '__main__':
    url = 'https://ned.ipac.caltech.edu/level5/Glossary/Essay_barnes_disk.html'
    texts = getWebContent(url)
    print(texts)
