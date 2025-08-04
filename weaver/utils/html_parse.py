import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter

from weaver.utils.config import config


def getWebContent(url):

    response = requests.get(url)
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
