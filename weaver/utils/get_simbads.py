import json
import re

import requests

# object_name="PSR J1959+2048"
def query_astronomical_object(object_name):
    # 设置查询 URL
    url = "https://simbad.cds.unistra.fr/simbad/sim-id"

    # 设置查询参数
    params = {
        "Ident": object_name,
        "output.format": "ASCII",  # 指定输出格式为 VOTable
    }

    # 发送 GET 请求
    response = requests.get(url, params=params)

    data = response.text
    return data


def parse_coordinates(line):
    parts = line.split(':')
    temp_part = parts[0].split('(')
    temp_part_part = temp_part[1].split(',')
    frame = temp_part_part[0].strip()
    ep = temp_part_part[1].strip().replace("ep=", "")
    eq = temp_part_part[2].strip().replace("eq=", "").replace(")", "")
    # 找到第一个部分的结束位置 (最后一个数字之前的空格)
    end_part1 = parts[1].find('(')

    # 找到第二个部分的结束位置 (最后一个 "]" 之后的空格)
    end_part2 = parts[1].rfind(']') + 1

    # 使用切片获取三部分
    value = parts[1][:end_part1].strip()
    if end_part1 != -1:
        notes = parts[1][end_part1:end_part2].strip()
        ref = parts[1][end_part2:].strip()
        return {
            "frame": frame,
            "ep": ep.strip(),
            "eq": eq.strip(),
            "value": value,
            "notes": notes,
            "ref": ref
        }
    else:
        return {
            "frame": frame,
            "ep": ep.strip(),
            "eq": eq.strip(),
            "value": value
        }


def parse_line(line):
    if 'Spectral type' not in line:
        key_value = line.split(':')
        key = key_value[0].strip().replace('(', '').replace(')', '')
        value = key_value[1].strip()
    else:
        key_value = line.split(':')
        key = key_value[0].strip().replace('(', '').replace(')', '')
        if 2 in key_value:
            value = key_value[1].strip() + ":" + key_value[2].strip()
        else:
            value = key_value[1].strip()
    return key, value


def get_Identifiers(text):
    identifiers = []
    # 提取 identifiers
    identifiers_match = re.search(r'Identifiers \(\d+\):\s+([\s\S]*?)\s+Bibcodes', text)
    if identifiers_match:
        identifiers_text = identifiers_match.group(1)
        identifiers = re.split(r'\s{2,}', identifiers_text.strip())
        identifiers = [id.strip() for id in identifiers if id.strip()]
    else:
        print("Identifiers section not matched.")
        identifiers = []
    return identifiers


def get_bibcodes(text):
    bibcodes = []
    # 提取 bibcodes
    bibcodes_match = re.search(r'Bibcodes\s+\d+-\d+ \(\) \(\d+\):\s+([\s\S]*?)\s+Measures', text)
    if bibcodes_match:
        bibcodes_text = bibcodes_match.group(1)
        bibcodes = re.split(r'\s{2,}', bibcodes_text.strip())
        bibcodes = [code.strip() for code in bibcodes if code.strip()]
    else:
        print("Bibcodes section not matched.")
        bibcodes = []
    return bibcodes


def get_simbad_data(obj_name):
    data = query_astronomical_object(obj_name)
    lines = data.split('\n')

    object_info = {}
    for line in lines:
        if 'Object' in line:
            # 定义正则表达式
            name_pattern = r'Object\s+(.*?)\s+---'
            type_pattern = r'---\s+(.*?)\s+---'
            id_pattern = r'OID=(.*?)\s+'

            # 使用正则表达式提取信息
            name_match = re.search(name_pattern, line)
            type_match = re.search(type_pattern, line)
            id_match = re.search(id_pattern, line)

            # 提取匹配到的内容
            name = name_match.group(1) if name_match else None
            type_ = type_match.group(1) if type_match else None
            id_ = id_match.group(1) if id_match else None
            object_info['name'] = name
            object_info['type'] = type_
            object_info['ID'] = id_

        elif 'Coordinates' in line:
            coordinate_data = parse_coordinates(line)
            frame = coordinate_data.pop("frame")
            if frame not in object_info:
                object_info[frame] = {}
            object_info[frame] = coordinate_data

        elif 'hierarchy counts' in line:
            key, value = parse_line(line)
            # 去掉每个字段中的前导字符（如#）
            cleaned_value = [v.split('=')[1].strip() for v in value.split(',')]
            # 将每个值转换为整数
            parents, children, siblings = map(int, cleaned_value)
            object_info[key] = {'parents': parents, 'children': children, 'siblings': siblings}

        elif 'Proper motions' in line:

            key, value = parse_line(line)
            key += "(mas/yr)"
            # 去除空格
            value = value.replace(' ', '')
            # 找到方括号的位置
            start_idx = value.index('[')
            end_idx = value.index(']') + 1
            # 分段
            zhi = value[:start_idx]
            notes = value[start_idx:end_idx + 1]
            ref = value[end_idx + 1:]
            object_info[key] = {'value': zhi, 'notes': notes, 'ref': ref}

        elif 'Parallax' in line or 'Radial Velocity' in line or 'Redshift' in line or 'cz' in line:

            key, value = parse_line(line)
            if key == "Parallax":
                key += "(mas)"
            # 以第一个空格为分隔符分为两部分
            first_split = value.split(' ', 1)

            # 再以第二部分中的第二个空格为分隔符再分为两部分
            second_split = first_split[1].split(' ', 1)

            # 获取三段内容
            zhi = first_split[0]
            notes = second_split[0] + ' ' + second_split[1][0]
            ref = second_split[1][1:]
            object_info[key] = {'value': zhi, 'notes': notes, 'ref': ref}

        elif 'Flux:' in line:
            band, value = line.split(':')
            # 以第一个空格为分隔符分为两部分
            first_split = value.split(' ', 2)

            # 再以第二部分中的第二个空格为分隔符再分为两部分
            second_split = first_split[2].split(' ', 2)

            # 获取三段内容
            zhi = first_split[1]
            notes = second_split[0] + ' ' + second_split[1][0]
            ref = second_split[2]
            object_info.setdefault('Fluxes', {})[band.strip()] = {'value': zhi, 'notes': notes, 'ref': ref}

        elif 'Spectral type' in line:
            key, value = parse_line(line)
            parts = value.split(' ')
            object_info[key] = {'value': parts[0], 'ref': parts[-1]}

        elif 'Morphological type' in line:
            key, value = parse_line(line)
            parts = value.split(' ')
            object_info[key] = {'value': parts[0], 'notes': parts[1], 'ref': parts[-1]}

        elif 'Angular size' in line:
            key, value = parse_line(line)
            # 找到第一个和第二个分隔符的索引
            first_space_idx = value.find('  ')
            second_space_idx = value.find(')') + 3

            # 进行分割
            part1 = value[:first_space_idx]
            part2 = value[first_space_idx + 2:second_space_idx + 1]
            part3 = value[second_space_idx + 2:]
            object_info[key] = {'value': part1, 'notes': part2, 'ref': part3}

        elif 'Identifiers' in line:
            object_info['identifiers'] = get_Identifiers(data)[:3]

        # elif 'Bibcodes' in line:
        #     object_info['bibcodes'] = get_bibcodes(data)

        elif 'Measures' in line:
            measures = lines[lines.index(line) + 1].strip()
            object_info['measures'] = measures

    # Convert to JSON
    json_data = json.dumps(object_info, indent=4)
    #print(json_data)
    json_data = json.loads(json_data)
    return json_data


#json_data = get_simbad_data("rigel")
#references = getReference(json_data["bibcodes"])
#print(references)
