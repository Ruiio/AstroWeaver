import re
import requests
import json
import logging
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse

from weaver.utils.config import config
from weaver.utils.html_parse_enhanced import EnhancedHTMLParser

logger = logging.getLogger(__name__)

class EnhancedWebSearcher:
    """增强的网络搜索器，提供更好的内容质量控制"""
    
    def __init__(self):
        self.html_parser = EnhancedHTMLParser()
        web_search_cfg = config.get("web_search", {}) if isinstance(config, dict) else {}
        self.open_websearch_url = web_search_cfg.get("open_websearch_url", "http://127.0.0.1:3001/mcp")
        self.open_websearch_engines = web_search_cfg.get("engines", ["duckduckgo", "bing"])
        
        # 排除的域名和关键词
        self.excluded_domains = {
            'pinterest.com', 'instagram.com', 'facebook.com', 'twitter.com',
            'linkedin.com', 'youtube.com', 'tiktok.com', 'reddit.com',
            'quora.com', 'yahoo.com', 'bing.com', 'google.com', 'dictionary.cambridge.org', 'iciba.com', 'collinsdictionary.com', 'merriam-webster.com', 'dictionary.com', 'vocabulary.com', 'wordreference.com'
        }
        
        self.excluded_keywords = {
            'pdf', 'abstract', 'articles', 'image', 'video', 'download',
            'login', 'register', 'subscribe', 'buy', 'shop', 'cart',
            'advertisement', 'ads', 'sponsored', 'promotion',
            'dictionary', '词典', '翻译', '是什么意思', 'iciba', 'youdao'
        }
        
        # 优先的域名（学术和科学网站）
        self.preferred_domains = {
            # 基础科学网站
            'wikipedia.org', 'arxiv.org', 'nature.com', 'science.org', 'aaas.org',
            
            # 天文组织和机构
            'nasa.gov', 'esa.int', 'iau.org', 'noao.edu', 'nrao.edu',
            'stsci.edu', 'spitzer.caltech.edu', 'hubblesite.org',
            'jwst.nasa.gov', 'chandra.harvard.edu', 'fermi.gsfc.nasa.gov',
            
            # 天文台和观测设施
            'keckobservatory.org', 'gemini.edu', 'vlt.eso.org', 'eso.org',
            'alma.cl', 'ligo.org', 'virgo-gw.eu', 'ska-telescope.org',
            'atnf.csiro.au', 'naic.edu', 'gb.nrao.edu',
            
            # 大学和研究机构
            'caltech.edu', 'harvard.edu', 'mit.edu', 'stanford.edu',
            'ox.ac.uk', 'cam.ac.uk', 'cern.ch', 'noaa.gov',
            'berkeley.edu', 'princeton.edu', 'yale.edu', 'columbia.edu',
            'uchicago.edu', 'arizona.edu', 'hawaii.edu',
            
            # 天文数据库和目录
            'simbad.cds.unistra.fr', 'vizier.cds.unistra.fr', 'ned.ipac.caltech.edu',
            'irsa.ipac.caltech.edu', 'heasarc.gsfc.nasa.gov', 'adc.gsfc.nasa.gov',
            'exoplanetarchive.ipac.caltech.edu', 'gaia.ari.uni-heidelberg.de',
            
            # 天文期刊和出版社
            'aanda.org', 'mnras.oxfordjournals.org', 'iopscience.iop.org',
            'apj.aas.org', 'aj.aas.org', 'pasp.aas.org',
            
            # 天文软件和工具
            'astropy.org', 'scipy.org', 'numpy.org', 'matplotlib.org',
            'skyandtelescope.org', 'astronomy.com', 'universetoday.com', 'space.com', 'timeanddate.com', 'earthsky.org',
            
            # 国际天文组织
            'cospar-assembly.org', 'iaus.org', 'astronomer.org'
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """检查URL是否有效和可接受"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # 检查是否在排除域名列表中
            if any(excluded in domain for excluded in self.excluded_domains):
                return False
            
            # 检查URL路径是否包含排除的关键词
            url_lower = url.lower()
            if any(keyword in url_lower for keyword in self.excluded_keywords):
                return False
            
            # 检查文件扩展名
            excluded_extensions = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'}
            if any(url_lower.endswith(ext) for ext in excluded_extensions):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"URL解析错误 {url}: {e}")
            return False
    
    def _score_url(self, url: str, title: str = "", snippet: str = "") -> float:
        """为URL评分，优先选择高质量来源"""
        score = 1.0
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # 优先域名加分
            if any(preferred in domain for preferred in self.preferred_domains):
                score += 2.0
            
            # 教育和政府域名加分
            if domain.endswith('.edu') or domain.endswith('.gov'):
                score += 1.5
            
            # 科学相关域名加分
            science_indicators = [
                'astro', 'space', 'cosmic', 'stellar', 'planet', 'galaxy', 'solar',
                'telescope', 'observatory', 'radio', 'optical', 'infrared', 'xray',
                'hubble', 'spitzer', 'chandra', 'kepler', 'jwst', 'gaia', 'alma',
                'keck', 'gemini', 'vlt', 'ligo', 'virgo', 'ska', 'nrao', 'eso',
                'simbad', 'vizier', 'ned', 'irsa', 'heasarc', 'exoplanet'
            ]
            if any(indicator in domain for indicator in science_indicators):
                score += 1.0
            
            # 标题和摘要质量评分
            content = (title + " " + snippet).lower()
            
            # 科学关键词加分
            science_keywords = [
                # 基础天文学科
                'astronomy', 'astrophysics', 'cosmology', 'astrometry', 'astrobiology',
                'stellar', 'galactic', 'planetary', 'extragalactic', 'interstellar',
                
                # 天体类型
                'star', 'planet', 'galaxy', 'nebula', 'supernova', 'pulsar', 'quasar',
                'black hole', 'neutron star', 'white dwarf', 'brown dwarf', 'exoplanet',
                'asteroid', 'comet', 'meteor', 'meteorite', 'solar system',
                
                # 天文现象
                'eclipse', 'transit', 'occultation', 'conjunction', 'opposition',
                'gravitational wave', 'gamma ray burst', 'supernova', 'nova',
                'variable star', 'binary star', 'multiple star', 'stellar evolution',
                
                # 观测和仪器
                'telescope', 'observatory', 'spectrometer', 'photometry', 'spectroscopy',
                'interferometry', 'radio astronomy', 'x-ray astronomy', 'infrared astronomy',
                'optical astronomy', 'space telescope', 'ground-based telescope',
                
                # 物理概念
                'redshift', 'parallax', 'luminosity', 'magnitude', 'flux', 'spectrum',
                'doppler effect', 'cosmic microwave background', 'dark matter', 'dark energy',
                'big bang', 'cosmic inflation', 'nucleosynthesis', 'accretion disk',
                
                # 研究方法
                'observation', 'research', 'study', 'analysis', 'discovery', 'scientific',
                'measurement', 'calibration', 'data reduction', 'modeling', 'simulation',
                'survey', 'catalog', 'database', 'archive'
            ]
            
            keyword_matches = sum(1 for keyword in science_keywords if keyword in content)
            score += keyword_matches * 0.2
            
            # 低质量指标减分
            low_quality_indicators = [
                'blog', 'forum', 'discussion', 'opinion', 'review',
                'buy', 'sell', 'price', 'cheap', 'discount'
            ]
            
            for indicator in low_quality_indicators:
                if indicator in content:
                    score -= 0.5
            
            return max(score, 0.1)  # 最低分数0.1
            
        except Exception as e:
            logger.warning(f"URL评分错误 {url}: {e}")
            return 0.5
    
    def _extract_top_links(self, search_data: Dict, top_n: int = 5) -> List[Dict]:
        """从搜索结果中提取高质量链接"""
        candidates = []
        
        # 提取organic结果
        if "organic" in search_data:
            for item in search_data["organic"]:
                url = item.get("link", "")
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                
                if self._is_valid_url(url):
                    score = self._score_url(url, title, snippet)
                    candidates.append({
                        'url': url,
                        'title': title,
                        'snippet': snippet,
                        'score': score
                    })
        
        # 按分数排序并返回前top_n个
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_n]
    
    def _extract_texts_from_links(self, links: List[Dict[str, Any]], max_results: int) -> List[dict]:
        """根据链接列表抓取正文文本。"""
        if not links:
            return []

        links.sort(key=lambda x: x.get('score', 0), reverse=True)
        top_links = links[:max_results]

        all_texts = []
        successful_extractions = 0

        for link_info in top_links:
            url = link_info.get('url', '')
            if not url:
                continue

            logger.info(f"正在处理URL: {url} (评分: {link_info.get('score', 0):.2f})")
            try:
                texts = self.html_parser.get_web_content(url)
                if texts:
                    for text in texts:
                        if isinstance(text, str) and len(text.strip()) > 50:
                            all_texts.append({
                                "text": text,
                                "source_type": "web",
                                "source_url": url,
                                "page_title": link_info.get('title', ''),
                                "search_engine": link_info.get('engine', '')
                            })
                    successful_extractions += 1
                    logger.info(f"成功从 {url} 提取 {len(texts)} 个文本块")
                else:
                    logger.warning(f"从 {url} 未提取到内容")
            except Exception as e:
                logger.error(f"处理URL {url} 时出错: {e}")
                continue

        # 兜底：若正文抓取失败，回退使用标题+摘要，避免整阶段零产出
        if not all_texts:
            fallback_blocks = []
            for link_info in top_links:
                title = (link_info.get('title') or '').strip()
                snippet = (link_info.get('snippet') or '').strip()
                url = link_info.get('url', '')
                if len(snippet) >= 30:
                    fallback_blocks.append({
                        "text": f"{title}\n{snippet}",
                        "source_type": "web_snippet",
                        "source_url": url,
                        "page_title": title,
                        "search_engine": link_info.get('engine', '')
                    })
            if fallback_blocks:
                logger.warning(f"正文抓取为空，已使用搜索摘要兜底: {len(fallback_blocks)} 条")
                all_texts = fallback_blocks

        logger.info(f"搜索完成: 成功处理 {successful_extractions}/{len(top_links)} 个链接，获得 {len(all_texts)} 个文本块")
        return all_texts

    def _search_with_open_websearch(self, query: str, max_results: int = 5) -> List[dict]:
        """通过 open-webSearch MCP HTTP 接口执行搜索。"""
        initialize_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "astroweaver", "version": "0.1"}
            }
        }

        init_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }

        init_resp = requests.post(self.open_websearch_url, headers=init_headers, json=initialize_payload, timeout=20)
        init_resp.raise_for_status()

        session_id = init_resp.headers.get("mcp-session-id")
        if not session_id:
            raise RuntimeError("open-webSearch 未返回 mcp-session-id")

        call_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {
                    "query": f"{query} astronomy skywatching site:nasa.gov OR site:esa.int OR site:wikipedia.org OR site:timeanddate.com OR site:space.com OR site:earthsky.org",
                    "limit": max_results * 6,
                    "engines": self.open_websearch_engines
                }
            }
        }

        call_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id
        }

        call_resp = requests.post(self.open_websearch_url, headers=call_headers, json=call_payload, timeout=60)
        call_resp.raise_for_status()

        # MCP streamable HTTP 返回 SSE 格式：event: message\ndata: {...}
        body = call_resp.text

        # 兼容两种返回：
        # 1) 多行 SSE data: ...
        # 2) 单行 data: { ... }（其中包含带换行的 JSON 字符串）
        data_text = ""
        if "\ndata:" in body or body.strip().startswith("data:"):
            # 优先取最后一个 data: 后面的完整 JSON 文本
            idx = body.rfind("data:")
            if idx != -1:
                data_text = body[idx + len("data:"):].strip()
        else:
            data_text = body.strip()

        if not data_text:
            raise RuntimeError("open-webSearch 返回为空")

        # 某些情况下会有 event: message 前缀，做清理
        data_text = re.sub(r"^event:\s*message\s*", "", data_text).strip()

        rpc_data = json.loads(data_text)
        text_payload = rpc_data.get("result", {}).get("content", [{}])[0].get("text", "")
        if not text_payload:
            raise RuntimeError("open-webSearch 未返回搜索结果文本")

        parsed = json.loads(text_payload)
        raw_results = parsed.get("results", [])

        links = []
        for item in raw_results:
            url = item.get("url", "")
            if not url or not self._is_valid_url(url):
                continue
            title = item.get("title", "")
            snippet = item.get("description", "")
            links.append({
                "url": url,
                "title": title,
                "snippet": snippet,
                "engine": item.get("engine", ""),
                "score": self._score_url(url, title, snippet)
            })

        # 质量增强：优先保留天文/科学优质域名，若数量不足再回退全量
        preferred_links = []
        for link in links:
            try:
                domain = urlparse(link.get("url", "")).netloc.lower()
            except Exception:
                domain = ""
            if any(preferred in domain for preferred in self.preferred_domains) or domain.endswith('.edu') or domain.endswith('.gov'):
                preferred_links.append(link)

        if preferred_links:
            logger.info(
                f"open-webSearch 查询 '{query}' 返回 {len(links)} 个可用链接，其中优质域名 {len(preferred_links)} 个，优先使用优质域名"
            )
            return self._extract_texts_from_links(preferred_links, max_results)

        logger.info(f"open-webSearch 查询 '{query}' 返回 {len(links)} 个可用链接（未命中优质域名，回退全量）")
        return self._extract_texts_from_links(links, max_results)

    def _search_with_bocha(self, query: str, max_results: int = 5) -> List[dict]:
        """通过原 bocha API 执行搜索（兼容保留）。"""
        url = "https://api.bochaai.com/v1/web-search"
        payload = {
            "query": query,
            "summary": True,
            "count": max_results
        }
        headers = {
            'Authorization': f'Bearer {config["api_keys"].get("X_API_KEY", "")}',
            'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        response_data = response.json()
        if response_data.get("code") != 200 or "data" not in response_data:
            raise RuntimeError(f"bocha 返回错误: {response_data.get('msg', '未知错误')}")

        web_pages = response_data["data"].get("webPages", {}).get("value", [])
        links = []
        for page in web_pages:
            page_url = page.get("url", "")
            if page_url and self._is_valid_url(page_url):
                title = page.get("name", "")
                snippet = page.get("snippet", "")
                links.append({
                    "url": page_url,
                    "title": title,
                    "snippet": snippet,
                    "engine": "bocha",
                    "score": self._score_url(page_url, title, snippet)
                })

        logger.info(f"bocha 查询 '{query}' 返回 {len(links)} 个可用链接")
        return self._extract_texts_from_links(links, max_results)

    def execute_web_query(self, query: str, max_results: int = 5) -> List[dict]:
        """执行网络搜索并返回高质量文本内容（优先 open-webSearch）。"""
        try:
            return self._search_with_open_websearch(query, max_results)
        except Exception as e:
            logger.error(f"open-webSearch 搜索失败，回退 bocha: {e}")
            try:
                return self._search_with_bocha(query, max_results)
            except Exception as e2:
                logger.error(f"bocha 搜索也失败: {e2}")
                return []
    
    def search_with_fallback(self, primary_query: str, fallback_queries: List[str] = None, max_results: int = 5) -> List[str]:
        """带有备用查询的搜索"""
        # 首先尝试主要查询
        results = self.execute_web_query(primary_query, max_results)
        
        # 如果结果不足且有备用查询，尝试备用查询
        if len(results) < max_results // 2 and fallback_queries:
            logger.info(f"主要查询结果不足，尝试备用查询")
            
            for fallback_query in fallback_queries:
                additional_results = self.execute_web_query(fallback_query, max_results - len(results))
                results.extend(additional_results)
                
                if len(results) >= max_results:
                    break
        
        return results[:max_results]


# 保持向后兼容性的函数
def execute_web_query(query: str) -> List[dict]:
    """向后兼容的函数，使用增强的搜索器"""
    searcher = EnhancedWebSearcher()
    return searcher.execute_web_query(query)


if __name__ == "__main__":
    # 测试增强的搜索器
    searcher = EnhancedWebSearcher()
    query = "Galactic thin disk astronomy"
    
    print(f"搜索查询: {query}")
    results = searcher.execute_web_query(query, max_results=3)
    
    print(f"\n获得 {len(results)} 个结果:")
    for i, text in enumerate(results):
        print(f"\n结果 {i+1}:")
        print(text[:300] + "..." if len(text) > 300 else text)
