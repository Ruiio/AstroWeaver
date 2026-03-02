# astroWeaver/utils/text_utils.py

import re
import unicodedata


def sanitize_filename(name: str) -> str:
    """
    清理字符串，使其成为一个有效的文件名。

    - 移除无效字符。
    - 将空格和常见分隔符替换为下划线。
    - 转换为小写。
    - 限制长度。

    Args:
        name (str): 原始字符串。

    Returns:
        str: 清理后的、可用作文件名的字符串。
    """
    if not isinstance(name, str):
        name = str(name)

    # 1. 规范化Unicode字符，例如将 'α' 转换为 'a'
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')

    # 2. 转换为小写
    name = name.lower()

    # 3. 将所有非字母数字的字符序列替换为单个下划线
    #    这个正则表达式会匹配一个或多个不是字母或数字的字符
    name = re.sub(r'[^a-z0-9]+', '_', name)

    # 4. 移除可能出现在开头或结尾的下划线
    name = name.strip('_')

    # 5. 限制文件名长度
    max_len = 100
    if len(name) > max_len:
        name = name[:max_len]
        # 再次确保截断后不会以下划线结尾
        name = name.strip('_')

    # 6. 如果清理后为空，则返回一个默认名称
    if not name:
        return "unnamed_entity"

    return name


def normalize_text(text: str) -> str:
    """
    对文本进行通用规范化处理。

    - 移除多余的空格。
    - 处理换行符。

    Args:
        text (str): 原始文本。

    Returns:
        str: 规范化后的文本。
    """
    if not isinstance(text, str):
        return ""

    # 将多个空格/换行符/制表符替换为单个空格
    text = re.sub(r'\s+', ' ', text)

    return text.strip()