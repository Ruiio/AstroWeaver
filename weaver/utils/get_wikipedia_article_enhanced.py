import os
import re
import logging
from typing import List, Optional, Dict
import wikipediaapi

logger = logging.getLogger(__name__)

class EnhancedWikipediaExtractor:
    """增强的Wikipedia文章提取器，提供更好的内容质量控制"""
    
    def __init__(self, user_agent: str = "AstroWeaver/2.0", language: str = "en"):
        # 设置代理（如果需要）
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        
        self.wiki = wikipediaapi.Wikipedia(user_agent, language)
        
        # 需要排除的章节标题模式
        self.excluded_section_patterns = {
            r'references?$',
            r'external\s+links?$',
            r'see\s+also$',
            r'further\s+reading$',
            r'bibliography$',
            r'notes?$',
            r'citations?$',
            r'sources?$',
            r'gallery$',
            r'images?$',
            r'media$',
            r'popular\s+culture$',
            r'in\s+fiction$',
            r'trivia$',
            r'miscellaneous$',
            r'awards?$',
            r'honors?$',
            r'legacy$',
            r'memorials?$',
            r'commemoration$'
        }
        
        # 低质量内容指标
        self.low_quality_indicators = {
            'stub', 'disambiguation', 'redirect', 'may refer to',
            'is a disambiguation page', 'this article is a stub'
        }
        
        # 最小章节长度
        self.min_section_length = 50
        self.max_section_length = 5000
    
    def _is_excluded_section(self, section_title: str) -> bool:
        """检查章节是否应该被排除"""
        title_lower = section_title.lower().strip()
        
        for pattern in self.excluded_section_patterns:
            if re.match(pattern, title_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _clean_section_content(self, content: str) -> str:
        """清理章节内容"""
        if not content:
            return ""
        
        # 移除Wikipedia特有的标记
        content = re.sub(r'\{\{[^}]*\}\}', '', content)  # 移除模板
        content = re.sub(r'\[\[[^\]]*\|([^\]]*)\]\]', r'\1', content)  # 简化链接
        content = re.sub(r'\[\[([^\]]*)\]\]', r'\1', content)  # 简化链接
        content = re.sub(r'\[[^\]]*\]', '', content)  # 移除外部链接
        
        # 移除引用标记
        content = re.sub(r'<ref[^>]*>.*?</ref>', '', content, flags=re.DOTALL)
        content = re.sub(r'<ref[^>]*/?>', '', content)
        
        # 移除HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        
        # 清理公式中的多余换行符
        content = self._clean_formula_newlines(content)
        
        # 清理空白字符
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content

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

    def _is_quality_content(self, content: str) -> bool:
        """判断内容是否有质量"""
        if not content or len(content) < self.min_section_length:
            return False
        
        if len(content) > self.max_section_length:
            return False
        
        # 检查是否包含低质量指标
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in self.low_quality_indicators):
            return False
        
        # 检查字母字符比例
        letter_count = sum(1 for c in content if c.isalpha())
        if letter_count < len(content) * 0.6:  # 至少60%是字母
            return False
        
        # 检查句子结构
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) < 2:  # 至少包含2个有效句子
            return False
        
        return True
    
    def _extract_sections_recursive(self, section, prefix: str = '', sections_list: List[str] = None) -> None:
        """递归提取章节内容"""
        if sections_list is None:
            sections_list = []
        
        section_title = f"{prefix}{section.title}".strip()
        
        # 检查是否应该排除此章节
        if self._is_excluded_section(section_title):
            logger.debug(f"排除章节: {section_title}")
            return
        
        # 清理和验证章节内容
        content = self._clean_section_content(section.text)
        
        if self._is_quality_content(content):
            # 添加章节标题作为上下文
            if section_title and section_title != section.text[:50]:
                formatted_content = f"Section: {section_title}\n{content}"
            else:
                formatted_content = content
            
            sections_list.append(formatted_content)
            logger.debug(f"添加章节: {section_title} ({len(content)} 字符)")
        else:
            logger.debug(f"跳过低质量章节: {section_title}")
        
        # 递归处理子章节
        for subsection in section.sections:
            self._extract_sections_recursive(
                subsection, 
                prefix=f"{section_title} > " if section_title else "", 
                sections_list=sections_list
            )
    
    def get_article_sections(self, title: str) -> List[str]:
        """获取Wikipedia文章的高质量章节内容"""
        logger.info(f"获取Wikipedia文章: '{title}'")
        
        try:
            page = self.wiki.page(title)
            
            if not page.exists():
                logger.warning(f"文章 '{title}' 不存在")
                return []
            
            sections = []
            
            # 处理文章摘要
            summary = self._clean_section_content(page.summary)
            if self._is_quality_content(summary):
                sections.append(f"Summary\n{summary}")
                logger.debug(f"添加摘要 ({len(summary)} 字符)")
            
            # 处理所有章节
            for section in page.sections:
                self._extract_sections_recursive(section, sections_list=sections)
            
            logger.info(f"从文章 '{title}' 提取了 {len(sections)} 个高质量章节")
            return sections
            
        except Exception as e:
            logger.error(f"获取文章 '{title}' 时出错: {e}")
            return []
    
    def get_article_info(self, title: str) -> Optional[Dict]:
        """获取文章的基本信息"""
        try:
            page = self.wiki.page(title)
            
            if not page.exists():
                return None
            
            return {
                'title': page.title,
                'url': page.fullurl,
                'summary_length': len(page.summary),
                'sections_count': len(page.sections),
                'exists': page.exists()
            }
            
        except Exception as e:
            logger.error(f"获取文章信息 '{title}' 时出错: {e}")
            return None
    
    def search_articles(self, query: str, limit: int = 5) -> List[str]:
        """搜索相关文章"""
        try:
            # 使用Wikipedia的搜索功能
            search_results = self.wiki.search(query, results=limit * 2)  # 获取更多结果以便过滤
            
            valid_articles = []
            for result in search_results:
                if len(valid_articles) >= limit:
                    break
                
                # 检查文章是否存在且有质量
                info = self.get_article_info(result)
                if info and info['exists'] and info['summary_length'] > 100:
                    valid_articles.append(result)
            
            logger.info(f"搜索 '{query}' 找到 {len(valid_articles)} 个有效文章")
            return valid_articles
            
        except Exception as e:
            logger.error(f"搜索文章 '{query}' 时出错: {e}")
            return []


# 保持向后兼容性的函数
def get_article_sections(title: str) -> List[str]:
    """向后兼容的函数，使用增强的提取器"""
    extractor = EnhancedWikipediaExtractor()
    return extractor.get_article_sections(title)


if __name__ == "__main__":
    # 测试增强的提取器
    extractor = EnhancedWikipediaExtractor()
    
    test_articles = ["Betelgeuse", "Milky Way", "Solar System"]
    
    for article in test_articles:
        print(f"\n{'='*50}")
        print(f"测试文章: {article}")
        print(f"{'='*50}")
        
        # 获取文章信息
        info = extractor.get_article_info(article)
        if info:
            print(f"文章信息: {info}")
        
        # 获取章节内容
        sections = extractor.get_article_sections(article)
        print(f"\n提取到 {len(sections)} 个章节:")
        
        for i, section in enumerate(sections[:3]):  # 只显示前3个章节
            print(f"\n章节 {i+1}:")
            print(section[:300] + "..." if len(section) > 300 else section)