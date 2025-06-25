import os

import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('MyProject', 'en')

def get_article_sections(title):
    """
    获取维基百科文章内容并按章节提取。
    """
    # 设置代理环境变量
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    page = wiki_wiki.page(title)
    if not page.exists():
        print(f"Article '{title}' does not exist.")
        return []

    sections = []

    main_content = page.summary.strip()
    if main_content:
        sections.append({
            "title": "Introduction",
            "content": main_content
        })

    def extract_sections(section, prefix='', sections=None):
        section_title = f"{prefix}{section.title}"
        if "References" in section_title or "External links" in section_title:
            return

        content = section.text.strip()
        if content:
            sections.append({
                "title": section_title,
                "content": content
            })

        for subsection in section.sections:
            extract_sections(subsection, prefix=section_title + " > ", sections=sections)

    for section in page.sections:
        extract_sections(section, sections=sections)

    return sections
# print(get_article_sections("Mars"))