import requests
import re
from bs4 import BeautifulSoup, Comment, NavigableString
from langchain_text_splitters import RecursiveCharacterTextSplitter
from weaver.utils.config import config
import logging

logger = logging.getLogger(__name__)

class EnhancedHTMLParser:
    """增强的HTML解析器，专注于提高数据质量"""
    
    def __init__(self):
        self.excluded_tags = {
            'script', 'style', 'nav', 'header', 'footer', 'aside', 
            'advertisement', 'ads', 'sidebar', 'menu', 'breadcrumb'
        }
        self.excluded_classes = {
            'advertisement', 'ads', 'sidebar', 'menu', 'nav', 'footer',
            'header', 'breadcrumb', 'social', 'share', 'comment',
            'popup', 'modal', 'overlay', 'cookie', 'gdpr'
        }
        self.excluded_ids = {
            'advertisement', 'ads', 'sidebar', 'menu', 'nav', 'footer',
            'header', 'comments', 'social', 'share'
        }
        
    def _is_excluded_element(self, element):
        """检查元素是否应该被排除"""
        # 检查元素是否为None或没有name属性
        if not element or not hasattr(element, 'name') or not element.name:
            return False
            
        if element.name in self.excluded_tags:
            return True
            
        # 检查class属性
        element_classes = element.get('class')
        if element_classes:
            classes = [cls.lower() for cls in element_classes]
            for exc_cls in self.excluded_classes:
                if exc_cls in ' '.join(classes):
                    return True
                
        # 检查id属性
        element_id = element.get('id')
        if element_id:
            element_id = element_id.lower()
            for exc_id in self.excluded_ids:
                if exc_id in element_id:
                    return True
                
        return False
    
    def _clean_text_content(self, text):
        """清理文本内容，移除多余空白和无用字符"""
        if not text:
            return ""
            
        # 移除HTML实体
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除常见的无用文本模式
        useless_patterns = [
            r"'Just a moment.*?",
            r"To ensure we keep this website safe.*?",
            r"COOKIES.*?",
            r"Please enable JavaScript.*?",
            r"This site requires JavaScript.*?",
            r"Loading.*?",
            r"Advertisement.*?",
            r"Skip to.*?",
            r"Click here.*?",
            r"Read more.*?",
            r"Continue reading.*?",
            r"\[.*?\]",  # 移除方括号内容
            r"\(.*?\)",  # 移除括号内容（如果太长）
        ]
        
        for pattern in useless_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # 移除过短或过长的括号内容
        text = re.sub(r'\([^)]{1,3}\)', '', text)  # 移除1-3字符的括号内容
        text = re.sub(r'\([^)]{100,}\)', '', text)  # 移除超长括号内容
        
        # 清理标点符号
        text = re.sub(r'[.,;:!?]{2,}', '.', text)  # 多个标点符号替换为单个句号
        text = re.sub(r'\s*[.,;:!?]\s*', '. ', text)  # 标准化标点符号间距
        
        # 移除多余空格和换行
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_meaningful_content(self, soup):
        """提取有意义的内容"""
        # 优先提取主要内容区域
        main_content_selectors = [
            'main', 'article', '.content', '.main-content', 
            '.article-content', '.post-content', '#content',
            '.entry-content', '.page-content', '#mw-content-text'
        ]
        
        main_content = None
        for selector in main_content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # 如果找不到主要内容区域，使用整个body
        if not main_content:
            main_content = soup.find('body') or soup
        
        # 在主要内容区域内清理不需要的元素
        if main_content:
            try:
                # 只清理主要内容区域内的不需要元素
                elements_to_remove = []
                for element in main_content.find_all():
                    if self._is_excluded_element(element):
                        elements_to_remove.append(element)
                
                # 安全地移除元素
                for element in elements_to_remove:
                    try:
                        element.decompose()
                    except Exception as e:
                        logger.debug(f"移除元素时出错: {e}")
            except Exception as e:
                logger.warning(f"清理主要内容区域时出错: {e}")
        
        # 提取段落和标题
        try:
            content_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
        except Exception as e:
            logger.warning(f"查找内容元素时出错: {e}")
            content_elements = []
        
        # 提取段落和标题
        try:
            content_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
        except Exception as e:
            logger.warning(f"查找内容元素时出错: {e}")
            content_elements = []
        
        meaningful_texts = []
        for element in content_elements:
            try:
                if not element or not hasattr(element, 'get_text'):
                    continue
                    
                text = element.get_text(separator=' ', strip=True)
                if not text:
                    continue
                    
                cleaned_text = self._clean_text_content(text)
                
                # 过滤太短或太长的文本
                if 10 <= len(cleaned_text) <= 2000:
                    # 检查文本质量
                    if self._is_meaningful_text(cleaned_text):
                        meaningful_texts.append(cleaned_text)
            except Exception as e:
                continue
        return meaningful_texts
    
    def _is_meaningful_text(self, text):
        """判断文本是否有意义"""
        if not text or len(text.strip()) < 10:
            return False
        
        # 检查是否包含足够的字母字符
        letter_count = sum(1 for c in text if c.isalpha())
        letter_ratio = letter_count / len(text) if len(text) > 0 else 0
        if letter_ratio < 0.5:  # 至少50%是字母
            return False
        
        # 检查是否包含常见的无用模式
        useless_indicators = [
            'cookie', 'gdpr', 'privacy policy', 'terms of service',
            'subscribe', 'newsletter', 'advertisement', 'sponsored',
            'loading', 'please wait', 'error', '404', '403', '500'
        ]
        
        text_lower = text.lower()
        for indicator in useless_indicators:
            if indicator in text_lower:
                return False
        
        # 检查重复字符或单词
        words = text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.7:  # 至少70%的单词是唯一的
                return False
        
        return True
    
    def get_web_content(self, url):
        """获取并解析网页内容"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"URL {url} 不是HTML内容: {content_type}")
                return []
            
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取有意义的内容
            meaningful_texts = self._extract_meaningful_content(soup)
            
            if not meaningful_texts:
                logger.warning(f"从URL {url} 未提取到有意义的内容")
                return []
            
            # 使用文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config['chunk']['chunk_size'],
                chunk_overlap=config['chunk']['chunk_overlap'],
                length_function=len,
            )
            
            # 合并所有有意义的文本
            combined_text = ' '.join(meaningful_texts)
            texts = text_splitter.create_documents([combined_text])
            
            # 进一步过滤分割后的文本块
            filtered_texts = []
            for item in texts:
                content = item.page_content.strip()
                if self._is_meaningful_text(content):
                    filtered_texts.append(content)
            
            logger.info(f"从URL {url} 成功提取 {len(filtered_texts)} 个高质量文本块")
            return filtered_texts
            
        except requests.exceptions.RequestException as e:
            logger.error(f"请求URL {url} 时出错: {e}")
            return []
        except Exception as e:
            logger.error(f"解析URL {url} 时出错: {e}")
            return []


# 保持向后兼容性的函数
def getWebContent(url):
    """向后兼容的函数，使用增强的解析器"""
    parser = EnhancedHTMLParser()
    return parser.get_web_content(url)


if __name__ == '__main__':
    # 测试增强的解析器
    parser = EnhancedHTMLParser()
    url = 'https://ned.ipac.caltech.edu/level5/Glossary/Essay_barnes_disk.html'
    texts = parser.get_web_content(url)
    
    print(f"提取到 {len(texts)} 个文本块:")
    for i, text in enumerate(texts[:3]):  # 只显示前3个
        print(f"\n文本块 {i+1}:")
        print(text[:200] + "..." if len(text) > 200 else text)