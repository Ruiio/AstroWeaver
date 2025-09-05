import re
import logging
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup, Tag, NavigableString
import requests
from .get_wikipedia_article_enhanced import EnhancedWikipediaExtractor

logger = logging.getLogger(__name__)

class EnhancedWikipediaClient:
    """增强的Wikipedia客户端，提供更好的内容提取和质量控制"""
    
    def __init__(self, user_agent: str = "AstroWeaver/2.0", language: str = "en"):
        self.extractor = EnhancedWikipediaExtractor(user_agent, language)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # 设置代理
        self.session.proxies = {
            'http': 'http://127.0.0.1:7890',
            'https': 'http://127.0.0.1:7890'
        }
        
        # 需要排除的HTML标签和属性
        self.excluded_tags = {
            'script', 'style', 'meta', 'link', 'noscript', 'template',
            'sup', 'sub'  # 上标和下标通常是引用
        }
        
        self.excluded_classes = {
            'reference', 'citation', 'navbox', 'infobox-below',
            'printfooter', 'catlinks', 'mw-editsection', 'noprint',
            'metadata', 'dablink', 'rellink', 'boilerplate',
            'ambox', 'tmbox', 'ombox', 'imbox', 'cmbox', 'fmbox'
        }
        
        self.excluded_ids = {
            'coordinates', 'siteSub', 'contentSub', 'jump-to-nav',
            'mw-navigation', 'footer', 'catlinks', 'references'
        }
    
    def _should_exclude_element(self, element: Tag) -> bool:
        """判断是否应该排除某个HTML元素"""
        if not isinstance(element, Tag):
            return False
        
        # 检查标签名
        if element.name in self.excluded_tags:
            return True
        
        # 检查class属性
        element_classes = element.get('class', [])
        if any(cls in self.excluded_classes for cls in element_classes):
            return True
        
        # 检查id属性
        element_id = element.get('id', '')
        if element_id in self.excluded_ids:
            return True
        
        # 检查style属性（隐藏元素）
        style = element.get('style', '')
        if 'display:none' in style.replace(' ', '') or 'visibility:hidden' in style.replace(' ', ''):
            return True
        
        return False
    
    def _extract_text_from_element(self, element, preserve_structure: bool = False) -> str:
        """从HTML元素中提取文本，改进的版本"""
        if isinstance(element, NavigableString):
            text = str(element).strip()
            # 过滤掉注释和其他非文本内容
            if text.startswith('<!--') or not text:
                return ''
            return text
        
        if not isinstance(element, Tag):
            return ''
        
        # 检查是否应该排除此元素
        if self._should_exclude_element(element):
            return ''
        
        text_parts = []
        
        for child in element.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text and not text.startswith('<!--'):
                    text_parts.append(text)
            elif isinstance(child, Tag):
                # 递归处理子元素
                child_text = self._extract_text_from_element(child, preserve_structure)
                if child_text:
                    # 根据标签类型添加适当的分隔符
                    if child.name in ['br']:
                        text_parts.append('\n')
                    elif child.name in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        if preserve_structure:
                            text_parts.append(f'\n{child_text}\n')
                        else:
                            text_parts.append(child_text)
                    elif child.name in ['li']:
                        if preserve_structure:
                            text_parts.append(f'• {child_text}\n')
                        else:
                            text_parts.append(child_text)
                    else:
                        text_parts.append(child_text)
        
        # 合并文本并清理
        full_text = ' '.join(text_parts)
        
        # 清理文本
        full_text = self._clean_extracted_text(full_text)
        
        return full_text
    
    def _clean_extracted_text(self, text: str) -> str:
        """清理提取的文本"""
        if not text:
            return ''
        
        # 移除方括号内的引用
        text = re.sub(r'\[[0-9\s,\-]+\]', '', text)
        
        # 移除坐标信息
        text = re.sub(r'\d+°\d+′[\d″′°NSEW\s]*', '', text)
        
        # 移除Wikipedia特有的标记
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        
        # 移除多余的括号内容（如果太长或包含特定模式）
        def clean_parentheses(match):
            content = match.group(1)
            # 如果括号内容太长或包含特定模式，则移除
            if (len(content) > 100 or 
                re.search(r'\d{4}[-–]\d{4}', content) or  # 年份范围
                'pronunciation' in content.lower() or
                'listen' in content.lower() or
                content.count(';') > 2):  # 太多分号
                return ''
            return match.group(0)
        
        text = re.sub(r'\(([^)]+)\)', clean_parentheses, text)
        
        # 清理公式中的多余换行符
        text = self._clean_formula_newlines(text)
        
        # 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # 移除行首行尾空白
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        return text.strip()

    def _clean_formula_newlines(self, text: str) -> str:
        """清理公式中的多余换行符，增强版本处理LaTeX格式"""
        if not text:
            return text
        
        # 1. 处理LaTeX displaystyle公式块
        # 匹配整个LaTeX公式块并清理内部换行
        def clean_latex_block(match):
            latex_content = match.group(1)
            # 移除LaTeX内部的所有换行和多余空白
            latex_content = re.sub(r'\s*\n\s*', ' ', latex_content)
            latex_content = re.sub(r'\s+', ' ', latex_content)
            return '{\\displaystyle ' + latex_content.strip() + '}'
        
        text = re.sub(r'\{\\\\displaystyle\s+([^}]+)\}', clean_latex_block, text)
        
        # 2. 处理LaTeX公式前后的换行和空白
        # 移除公式前后的多余换行
        text = re.sub(r'\n\s*\n\s*\{\\\\displaystyle', '\n{\\\\displaystyle', text)
        text = re.sub(r'\}\s*\n\s*\n', '}\n', text)
        
        # 3. 处理普通数学表达式中的换行符
        # 处理等号周围的换行
        text = re.sub(r'\s*\n\s*=\s*\n\s*', ' = ', text)
        text = re.sub(r'\s*\n\s*=\s*', ' = ', text)
        text = re.sub(r'\s*=\s*\n\s*', ' = ', text)
        
        # 处理运算符周围的换行
        text = re.sub(r'\s*\n\s*([+\-*/≥≤])\s*\n\s*', r' \1 ', text)
        text = re.sub(r'\s*\n\s*([+\-*/≥≤])\s*', r' \1 ', text)
        text = re.sub(r'\s*([+\-*/≥≤])\s*\n\s*', r' \1 ', text)
        
        # 处理指数符号周围的换行
        text = re.sub(r'\s*\n\s*\^\s*\n\s*', '^', text)
        text = re.sub(r'\s*\n\s*\^\s*', '^', text)
        text = re.sub(r'\s*\^\s*\n\s*', '^', text)
        
        # 处理括号内的换行
        text = re.sub(r'\(\s*\n\s*', '(', text)
        text = re.sub(r'\s*\n\s*\)', ')', text)
        
        # 处理变量和数字之间的换行
        text = re.sub(r'([a-zA-Z0-9])\s*\n\s*([a-zA-Z0-9])', r'\1\2', text)
        
        # 处理函数名和括号之间的换行
        text = re.sub(r'([a-zA-Z]+)\s*\n\s*\(', r'\1(', text)
        
        # 处理分数和根号等数学符号
        text = re.sub(r'sqrt\s*\n\s*\(', 'sqrt(', text)
        text = re.sub(r'log\s*\n\s*\(', 'log(', text)
        text = re.sub(r'sin\s*\n\s*\(', 'sin(', text)
        text = re.sub(r'cos\s*\n\s*\(', 'cos(', text)
        text = re.sub(r'tan\s*\n\s*\(', 'tan(', text)
        
        # 处理希腊字母和特殊符号
        text = re.sub(r'([πα-ωΑ-Ω])\s*\n\s*', r'\1 ', text)
        
        # 4. 处理数学公式块前后的多余空行
        # 移除公式前后的过多空行，但保留一个换行用于分隔
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 5. 清理数学符号周围的空白
        text = re.sub(r'\s*\\,\s*', ' ', text)  # LaTeX thin space
        text = re.sub(r'\s*\\;\s*', ' ', text)  # LaTeX medium space
        text = re.sub(r'\s*\\quad\s*', ' ', text)  # LaTeX quad space
        
        return text

    def get_article_sections(self, title: str) -> List[str]:
        """获取文章章节，使用增强的提取器"""
        return self.extractor.get_article_sections(title)
    
    def scrape_infobox(self, title: str) -> Dict[str, str]:
        """抓取Wikipedia页面的Infobox信息，增强版本"""
        logger.info(f"抓取Infobox: '{title}'")
        
        try:
            # 构建Wikipedia URL
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            
            # 发送请求
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # 解析HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 查找infobox
            infobox = soup.find('table', class_=re.compile(r'infobox', re.I))
            
            if not infobox:
                logger.warning(f"未找到Infobox: '{title}'")
                return {}
            
            infobox_data = {}
            
            # 提取标题
            title_element = infobox.find(['th', 'caption'], class_=re.compile(r'infobox-title|fn', re.I))
            if title_element:
                title_text = self._extract_text_from_element(title_element)
                if title_text:
                    infobox_data['title'] = title_text
            
            # 提取所有行
            rows = infobox.find_all('tr')
            
            for row in rows:
                # 查找标签和值
                label_cell = row.find(['th', 'td'], class_=re.compile(r'infobox-label', re.I))
                data_cell = row.find(['td'], class_=re.compile(r'infobox-data', re.I))
                
                # 如果没有找到特定的class，尝试通用方法
                if not label_cell or not data_cell:
                    cells = row.find_all(['th', 'td'])
                    if len(cells) >= 2:
                        label_cell, data_cell = cells[0], cells[1]
                    elif len(cells) == 1 and cells[0].name == 'th':
                        # 可能是子标题
                        continue
                
                if label_cell and data_cell:
                    # 提取标签文本
                    label = self._extract_text_from_element(label_cell)
                    
                    # 提取数据文本
                    data = self._extract_text_from_element(data_cell)
                    
                    # 验证和清理
                    if label and data and self._is_valid_infobox_entry(label, data):
                        # 标准化标签名
                        label = self._normalize_infobox_label(label)
                        infobox_data[label] = data
            
            logger.info(f"从 '{title}' 提取了 {len(infobox_data)} 个Infobox条目")
            return infobox_data
            
        except Exception as e:
            logger.error(f"抓取Infobox '{title}' 时出错: {e}")
            return {}
    
    def _is_valid_infobox_entry(self, label: str, data: str) -> bool:
        """验证Infobox条目是否有效"""
        if not label or not data:
            return False
        
        # 过滤掉太短或太长的条目
        if len(label) < 2 or len(label) > 100:
            return False
        
        if len(data) < 1 or len(data) > 1000:
            return False
        
        # 过滤掉无意义的条目
        meaningless_patterns = {
            r'^\s*[-–—]+\s*$',  # 只有破折号
            r'^\s*[?]+\s*$',    # 只有问号
            r'^\s*n/?a\s*$',    # N/A
            r'^\s*unknown\s*$', # Unknown
            r'^\s*tbd\s*$',     # TBD
        }
        
        data_lower = data.lower().strip()
        for pattern in meaningless_patterns:
            if re.match(pattern, data_lower, re.IGNORECASE):
                return False
        
        return True
    
    def _normalize_infobox_label(self, label: str) -> str:
        """标准化Infobox标签名"""
        # 移除特殊字符并转换为小写
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', label)
        normalized = re.sub(r'\s+', '_', normalized.strip().lower())
        
        # 标准化常见标签
        label_mappings = {
            'right_ascension': 'ra',
            'declination': 'dec',
            'apparent_magnitude': 'magnitude',
            'absolute_magnitude': 'abs_magnitude',
            'spectral_type': 'spectral_class',
            'surface_temperature': 'temperature',
        }
        
        return label_mappings.get(normalized, normalized)
    
    def get_enhanced_article_data(self, title: str) -> Dict:
        """获取文章的完整增强数据"""
        logger.info(f"获取增强文章数据: '{title}'")
        
        result = {
            'title': title,
            'sections': [],
            'infobox': {},
            'metadata': {}
        }
        
        # 获取基本信息
        info = self.extractor.get_article_info(title)
        if info:
            result['metadata'] = info
        
        # 获取章节内容
        sections = self.get_article_sections(title)
        result['sections'] = sections
        
        # 获取Infobox
        infobox = self.scrape_infobox(title)
        result['infobox'] = infobox
        
        logger.info(f"完成增强数据获取: {len(sections)} 章节, {len(infobox)} Infobox条目")
        return result


# 保持向后兼容性
class WikipediaClient(EnhancedWikipediaClient):
    """向后兼容的Wikipedia客户端"""
    pass


if __name__ == "__main__":
    # 测试增强的客户端
    client = EnhancedWikipediaClient()
    
    test_articles = ["Betelgeuse", "Andromeda Galaxy"]
    
    for article in test_articles:
        print(f"\n{'='*60}")
        print(f"测试文章: {article}")
        print(f"{'='*60}")
        
        # 获取完整数据
        data = client.get_enhanced_article_data(article)
        
        print(f"\n元数据: {data['metadata']}")
        print(f"\n章节数量: {len(data['sections'])}")
        print(f"Infobox条目数量: {len(data['infobox'])}")
        
        # 显示Infobox示例
        if data['infobox']:
            print("\nInfobox示例:")
            for key, value in list(data['infobox'].items())[:5]:
                print(f"  {key}: {value[:100]}{'...' if len(value) > 100 else ''}")
        
        # 显示章节示例
        if data['sections']:
            print("\n第一个章节示例:")
            first_section = data['sections'][0]
            print(first_section[:300] + "..." if len(first_section) > 300 else first_section)