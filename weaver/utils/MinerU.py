import requests
import os
import json


from weaver.utils.config import config


def parse_file(file_paths, server_url=config['minerU']['api_url'], **kwargs):
    """
    调用文件解析API，上传文件并获取解析结果。

    :param file_paths: 单个文件路径 (str) 或多个文件路径的列表 (list of str)。
    :param server_url: API的URL地址。
    :param kwargs: 其他API参数，例如 lang_list, return_md, table_enable 等。
                   这些参数会覆盖默认值。
    :return: 解析成功时返回API的JSON响应内容 (dict)，失败时返回 None。
    """
    # 如果传入的是单个文件路径，将其转换为列表以便统一处理
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # 准备 multipart/form-data 中的 'files' 部分
    # requests库要求文件部分是一个元组列表，格式为:
    # [('field_name', (filename, file_object, content_type)), ...]
    files_to_upload = []
    opened_files = []  # 用于后续关闭文件句柄
    try:
        for path in file_paths:
            if not os.path.exists(path):
                print(f"错误：文件不存在 -> {path}")
                return None
            # 'rb' 表示以二进制只读模式打开文件，适用于所有文件类型
            file_obj = open(path, 'rb')
            opened_files.append(file_obj)
            # 'files' 是接口要求的字段名
            files_to_upload.append(('files', (os.path.basename(path), file_obj)))
    except Exception as e:
        print(f"打开文件时出错: {e}")
        # 确保已打开的文件被关闭
        for f in opened_files:
            f.close()
        return None

    # 准备 multipart/form-data 中的 'data' 部分
    # 根据接口文档设置默认参数
    data_payload = {
        'lang_list': ['ch'],
        'return_md': True,
        'table_enable': True,
        'formula_enable': True,
        # 您可以根据需要添加或修改其他默认参数
        #'output_dir': r'./output_dir',
        # 'backend': 'pipeline',
        # 'parse_method': 'auto',
        # 'return_middle_json': False,
        # 'start_page_id': 0,
        # 'end_page_id': 99999,
    }

    # 使用用户传入的kwargs覆盖默认参数
    data_payload.update(kwargs)

    # 将布尔值转换为小写字符串 "true" / "false"
    for key, value in data_payload.items():
        if isinstance(value, bool):
            data_payload[key] = str(value).lower()

    print("准备发送pdf解析请求...")
    print(f"URL: {server_url}")
    print(f"文件: {[path for path in file_paths]}")
    print(f"参数: {data_payload}")

    try:
        # 发送POST请求
        response = requests.post(server_url, files=files_to_upload, data=data_payload, timeout=300)  # 设置300秒超时

        # 打印响应状态码和内容，帮助诊断问题
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text[:500]}...(截断)")

        # 检查HTTP响应状态码，如果不是2xx，则会抛出异常
        response.raise_for_status()
        result_json={}
        for item in response.json()['results']:
            result_json[item]=response.json()['results'][item]['md_content']

        # 解析并返回JSON响应
        return result_json

    except requests.exceptions.RequestException as e:
        print(f"请求API时发生错误: {e}")
        return None
    except json.JSONDecodeError:
        print("无法解析API返回的JSON。原始响应内容:")
        print(response.text)
        return None
    finally:
        # 无论成功与否，都确保关闭所有打开的文件
        for f in opened_files:
            f.close()
        print("所有文件句柄已关闭。")


# --- 主程序入口：如何使用上面的函数 ---
if __name__ == "__main__":
    # --- 准备一个用于测试的虚拟文件 ---
    # 在实际使用中，请将 DUMMY_FILE_PATH 替换为您自己的文件路径
    DUMMY_FILE_PATH = r"C:\Users\Administrator\Desktop\astroWeaver\data\input\test_data\our_solar_system_lithograph.pdf"


    # --- 示例1: 最简单的调用，使用默认参数解析单个文件 ---
    print("\n" + "=" * 20 + " 示例 1: 基本调用 " + "=" * 20)
    result_1 = parse_file(DUMMY_FILE_PATH)

    if result_1:
        print("\n✅ API调用成功！响应内容:")
        # 使用json.dumps美化输出，ensure_ascii=False以正确显示中文
        print(json.dumps(result_1, indent=2, ensure_ascii=False))
    else:
        print("\n❌ API调用失败。")
