# astroWeaver/data_sources/wikipedia_client.py

import logging
import re
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from weaver.utils.get_wikipeida_article import get_article_sections

logger = logging.getLogger(__name__)


# --- 从你的代码中完整迁移过来的Infobox抓取辅助函数 ---

def _split_camel_case(text):
    text = re.sub(r'(?<!^)(?<![A-Z\s])([A-Z])', r' \1', text)
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)', r' ', text)
    text = re.sub(r'(?<=\d)(?=[a-zA-Z])', r' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_and_join_text_parts(text_parts):
    value = ""
    for i, part in enumerate(text_parts):
        if part == "%%%BR_SEPARATOR%%%":
            if value and not value.endswith(", "): value += ", "
        else:
            if value and not value.endswith(tuple([", ", " ", "(", "[", "-"])) and \
                    not part.startswith(tuple([".", ",", ";", ":", ")", "]", "-"])): value += " "
            value += part
    value = value.strip()
    value = re.sub(r'\s*,\s*', ', ', value)
    value = re.sub(r'(,\s*)+', ', ', value)
    value = value.strip(', ')
    value = re.sub(r'[ \t]+', ' ', value).strip()
    value = re.sub(r'(\S)\(', r'\1 (', value)
    value = re.sub(r'\)(\S)', r') \1', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def _extract_text_from_element(element):
    if not element: return ""
    for s_tag in element.find_all(['style', 'script'], recursive=False): s_tag.decompose()
    for hidden_el in element.find_all(style=lambda s: s and 'display:none' in s.lower(),
                                      recursive=False): hidden_el.decompose()
    for noprint_el in element.find_all(class_='noprint', recursive=False): noprint_el.decompose()
    for listen_span in element.find_all('span', class_='IPA', recursive=False):
        if listen_span.find('span', title=lambda t: t and 'listen' in t.lower()): listen_span.decompose()
    for sup_tag in list(element.find_all('sup')):
        if not sup_tag.parent: continue
        is_reference_sup = (sup_tag.has_attr('class') and any(
            cls in sup_tag['class'] for cls in ['reference', 'Template-Fact', 'noprint', 'mw-ref'])) or \
                           (sup_tag.find('a', href=lambda x: x and x.startswith('#cite_note-'))) or \
                           (sup_tag.get_text(strip=True).startswith('['))
        if is_reference_sup: sup_tag.decompose(); continue
        sup_text_content = sup_tag.get_text(strip=True)
        is_numeric_exponent_candidate = bool(re.fullmatch(r'[+\-−]?\d+', sup_text_content))
        previous_element = sup_tag.previous_sibling
        if is_numeric_exponent_candidate:
            handled_sup = False
            if previous_element and isinstance(previous_element, NavigableString):
                text_of_previous_element = str(previous_element)
                text_to_match = text_of_previous_element.rstrip()
                match_base_10 = re.search(r'([×xXEe]?10)$', text_to_match)
                if match_base_10:
                    base_text = match_base_10.group(1)
                    is_standalone_10 = base_text == '10'
                    can_format_exponent = True
                    if is_standalone_10 and len(text_to_match) > len(base_text):
                        char_before_base = text_to_match[-(len(base_text) + 1)]
                        if not char_before_base.isspace() and char_before_base not in ['×', 'x', 'E',
                                                                                       'e']: can_format_exponent = False
                    if can_format_exponent:
                        part_before_base = text_to_match[:-len(base_text)]
                        formatted_base = "10^"
                        if base_text.lower().startswith('e') or base_text.lower().startswith(
                                'x') or base_text.lower().startswith('×'): formatted_base = base_text[0] + "10^"
                        new_text = part_before_base + formatted_base + sup_text_content + text_of_previous_element[
                                                                                          len(text_to_match):]
                        previous_element.replace_with(NavigableString(new_text));
                        sup_tag.decompose();
                        handled_sup = True
                elif not handled_sup and text_to_match and (text_to_match[-1].isalpha() or text_to_match[-1] == ')'):
                    new_text = text_to_match + "^" + sup_text_content + text_of_previous_element[len(text_to_match):]
                    previous_element.replace_with(NavigableString(new_text));
                    sup_tag.decompose();
                    handled_sup = True
            if not handled_sup: sup_tag.decompose()
        elif sup_text_content in ['²', '³']:
            formatted_sup = "^2" if sup_text_content == '²' else "^3"
            if previous_element and isinstance(previous_element, NavigableString):
                new_text = str(previous_element) + formatted_sup
                previous_element.replace_with(NavigableString(new_text))
            else:
                sup_tag.insert_before(NavigableString(formatted_sup))
            sup_tag.decompose()
        else:
            sup_tag.decompose()
    final_text_parts = []
    for child in element.children:
        if isinstance(child, NavigableString):
            text = str(child).strip();
            if text: final_text_parts.append(text)
        elif isinstance(child, Tag):
            if child.name == 'br':
                if final_text_parts and final_text_parts[-1] != "%%%BR_SEPARATOR%%%": final_text_parts.append(
                    "%%%BR_SEPARATOR%%%")
            elif child.name in ['ul', 'ol']:
                list_items = []
                for li in child.find_all('li', recursive=False):
                    li_text = _extract_text_from_element(li).lstrip('•').strip()
                    if li_text: list_items.append(li_text)
                if list_items: final_text_parts.append(", ".join(list_items))
            else:
                text = child.get_text(separator=' ', strip=True)
                if text: final_text_parts.append(text)
    cleaned_value = clean_and_join_text_parts(final_text_parts)
    cleaned_value = re.sub(r'\s*\[[a-zA-Z0-9\s-]+\]\s*', '', cleaned_value)
    cleaned_value = re.sub(r'\s*\(\s*listen\s*\)\s*', '', cleaned_value, flags=re.IGNORECASE)
    cleaned_value = re.sub(r'\s*Coordinates:.*', '', cleaned_value, flags=re.IGNORECASE)
    return cleaned_value.strip()


class WikipediaClient:
    """
    封装与维基百科的交互，包括获取文章章节和抓取Infobox。
    """

    def __init__(self, user_agent: str = "AstroWeaver/1.0 (https://github.com/your/repo; your.email@example.com)"):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        logger.info("WikipediaClient initialized.")

    def get_article_sections(self, entity_name: str) -> Optional[List]:
        """
        获取指定维基百科文章的所有章节及其内容。
        """
        logger.info(f"Fetching Wikipedia sections for: '{entity_name}'")
        try:
            # 调用你提供的函数
            sections = get_article_sections(entity_name)
            if not sections:
                logger.warning(f"No sections found for '{entity_name}'.")
                return []
            logger.info(f"Found {len(sections)} sections for '{entity_name}'.")
            url = f"https://en.wikipedia.org/wiki/{entity_name.replace(' ', '_')}"
            wrapped = []
            for s in sections:
                if isinstance(s, str) and s.strip():
                    wrapped.append({
                        "text": s,
                        "source_type": "wikipedia",
                        "source_url": url
                    })
            return wrapped
        except Exception as e:
            logger.error(f"Error fetching sections for '{entity_name}': {e}")
            return []

    def get_infobox(self, entity_name: str) -> Optional[Dict[str, str]]:
        """
        通过网页抓取来获取文章的Infobox内容。
        此函数完整迁移自你的参考代码。
        """
        logger.info(f"Scraping Infobox for: '{entity_name}'")
        url = f"https://en.wikipedia.org/wiki/{entity_name.replace(' ', '_')}"

        headers = {
            'User-Agent': 'MyKnowledgeGraphBuilder/1.3 (myemail@example.com; http://myproject.example.com)'}
        attributes = {}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            html_content = response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching page {entity_name} for infobox: {e}")
            return None
        soup = BeautifulSoup(html_content, 'lxml')
        infobox = soup.find('table', class_=lambda x: x and 'infobox' in x)
        if not infobox:
            logging.info(f"No infobox found for {entity_name} using web scraping.")
            return {}
        caption_tag = infobox.find('caption', recursive=False)
        if caption_tag:
            caption_text = _extract_text_from_element(caption_tag)
            # MODIFIED: Compare with original page_title
            if caption_text and caption_text.lower() != entity_name.lower() and len(caption_text.split()) < 5:
                attributes["Infobox Title"] = caption_text
        table_body = infobox.find('tbody')
        rows_container = table_body if table_body else infobox
        last_th_colspan_text = None
        for row in rows_container.find_all('tr', recursive=False):
            label_text, value_text = None, None
            header_colspan = row.find('th', attrs={'colspan': True}, recursive=False)
            if header_colspan:
                current_section_label_candidate = _extract_text_from_element(header_colspan)
                if current_section_label_candidate:
                    last_th_colspan_text = re.sub(r'\s*\[\w+\]$', '', current_section_label_candidate).strip()
                    if ' ' not in last_th_colspan_text and any(c.islower() for c in last_th_colspan_text) and any(
                            c.isupper() for c in last_th_colspan_text):
                        last_th_colspan_text = _split_camel_case(last_th_colspan_text)
                continue
            th_label = row.find('th', scope='row', recursive=False)
            if th_label:
                label_text = _extract_text_from_element(th_label)
                td_value = th_label.find_next_sibling('td')
                if td_value: value_text = _extract_text_from_element(td_value)
                last_th_colspan_text = None
            else:
                td_cells = row.find_all('td', recursive=False)
                if len(td_cells) == 2:
                    potential_label_text = _extract_text_from_element(td_cells[0])
                    is_label_like = len(potential_label_text.split()) < 5 and (td_cells[0].find('b') or (
                            td_cells[0].find('a') and not td_cells[0].find('a', class_="image")))
                    if is_label_like and potential_label_text:
                        label_text = potential_label_text
                        value_text = _extract_text_from_element(td_cells[1])
                        last_th_colspan_text = None
                elif len(td_cells) == 1 and last_th_colspan_text:
                    label_text = last_th_colspan_text
                    value_text = _extract_text_from_element(td_cells[0])
            if label_text:
                label_text = re.sub(r'\s+', ' ', label_text).strip()
                if not label_text: continue
                if ' ' not in label_text and any(c.islower() for c in label_text) and any(
                        c.isupper() for c in label_text): label_text = _split_camel_case(label_text)
                label_text = re.sub(r'\s*\[\w+\]$', '', label_text).strip()
                if value_text:
                    attributes[label_text] = value_text
                elif label_text and not value_text:
                    attributes[label_text] = ""
            if th_label or (len(td_cells) == 2 and is_label_like and potential_label_text): last_th_colspan_text = None
        logger.info(f"Found {len(attributes)} attributes for {entity_name}")
        return attributes


# 示例用法
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    client = WikipediaClient()

    # 测试获取章节
    mars_sections = client.get_article_sections("Betelgeuse")
    if mars_sections:
        print(f"\n--- Sections for Betelgeuse (found {len(mars_sections)}) ---")
        print(f"First section content snippet: {mars_sections[0][:100]}...")

    # 测试获取Infobox
    mars_infobox = client.get_infobox("Betelgeuse")
    if mars_infobox:
        print("\n--- Infobox for Betelgeuse ---")
        for key, value in list(mars_infobox.items())[:5]:  # 只打印前5个属性
            print(f"{key}: {value}")
