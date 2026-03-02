import json
import requests
import re


class SimpleWikidataClient:
    def __init__(self, proxies=None):
        self.proxies = proxies or {}
        self.headers = {
            'User-Agent': 'AstroWeaverBot/1.0 (Integrated Pipeline)'
        }

        self.session = requests.Session()
        self.session.proxies.update(self.proxies)
        self.session.headers.update(self.headers)

        # 缓存：存储 Q号(实体) 和 P号(属性) 的 Label
        self.label_cache = {}

        # 加载本地映射作为优先覆盖（可选）
        self.local_props_map = self._load_props_map()

    def _load_props_map(self):
        """加载本地属性映射文件"""
        try:
            with open('props_simple.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {item['id']: item['label'] for item in data}
        except FileNotFoundError:
            return {}

    def get_label_from_cache(self, code):
        """从缓存或本地映射中获取 Label"""
        # 1. 优先看本地文件有没有定义
        if code in self.local_props_map:
            return self.local_props_map[code]
        # 2. 看缓存有没有
        return self.label_cache.get(code, code)

    def _batch_fetch_labels(self, codes):
        """
        批量获取 Label (同时支持 Q号 和 P号)
        """
        # 过滤掉已经缓存的 ID，只请求未知的
        # 正则匹配 Q开头或P开头的数字 ID
        to_fetch = list(set([
            str(c) for c in codes
            if str(c) not in self.label_cache
               and str(c) not in self.local_props_map
               and re.match(r'^[QP]\d+$', str(c))
        ]))

        if not to_fetch:
            return

        chunk_size = 50
        url = 'https://www.wikidata.org/w/api.php'

        for i in range(0, len(to_fetch), chunk_size):
            chunk = to_fetch[i:i + chunk_size]
            ids_str = "|".join(chunk)

            params = {
                'action': 'wbgetentities',
                'ids': ids_str,
                'format': 'json',
                'props': 'labels',
                'languages': 'en'
            }

            try:
                res = self.session.get(url, params=params, timeout=10)
                data = res.json()
                entities = data.get('entities', {})

                for eid, entity in entities.items():
                    label = entity.get('labels', {}).get('en', {}).get('value', eid)
                    self.label_cache[eid] = label

            except Exception as e:
                print(f"[Wikidata] Batch fetch error: {e}")

    def search(self, query, limit=1):
        url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities', 'format': 'json', 'search': query,
            'language': 'en', 'type': 'item', 'limit': limit
        }
        try:
            res = self.session.get(url, params=params, timeout=5)
            data = res.json()
            return data.get("search", [])
        except Exception as e:
            print(f"[Wikidata] Search error: {e}")
            return []

    def process_entity(self, query_name):
        search_res = self.search(query_name)
        if not search_res:
            return None

        qcode = search_res[0]['id']
        url = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbgetentities', 'ids': qcode, 'format': 'json',
            'props': 'labels|descriptions|claims|aliases', 'languages': 'en'
        }

        try:
            res = self.session.get(url, params=params, timeout=10)
            data = res.json()
            if 'entities' not in data or qcode not in data['entities']:
                return None

            result = data['entities'][qcode]
            claims = result.get('claims', {})

            # --- 关键修改: 收集所有 ID (包括属性 P 和 值 Q) ---
            ids_to_prefetch = set()

            # 1. 收集属性 ID (P-codes)
            for prop_id in claims.keys():
                ids_to_prefetch.add(prop_id)

            # 2. 收集值 ID (Q-codes)
            for prop_id, claim_list in claims.items():
                for item in claim_list:
                    mainsnak = item.get('mainsnak', {})
                    dt = mainsnak.get('datatype')
                    dv = mainsnak.get('datavalue', {})
                    if not dv: continue

                    if dt == "wikibase-item":
                        val_id = dv.get('value', {}).get('id')
                        if val_id: ids_to_prefetch.add(val_id)
                    elif dt == "quantity":
                        unit_url = dv.get('value', {}).get('unit', '')
                        if "entity/" in unit_url:
                            ids_to_prefetch.add(unit_url.split('/')[-1])

            # --- 批量查询所有 Label ---
            if ids_to_prefetch:
                self._batch_fetch_labels(list(ids_to_prefetch))

            # --- 构建结果 ---
            entity_data = {
                "Qcode": qcode,
                "label": result.get('labels', {}).get('en', {}).get('value', ''),
                "description": result.get('descriptions', {}).get('en', {}).get('value', ''),
                "aliases": ", ".join([a['value'] for a in result.get('aliases', {}).get('en', [])])
            }

            for prop_id, claim_list in claims.items():
                # 获取属性的英文名 (现在缓存里肯定有了)
                prop_label = self.get_label_from_cache(prop_id)

                for item in claim_list:
                    mainsnak = item.get('mainsnak', {})
                    if not mainsnak: continue

                    dt = mainsnak.get('datatype')
                    dv = mainsnak.get('datavalue', {})
                    if not dv: continue

                    val = dv.get('value')
                    final_val = ""

                    if dt == "wikibase-item":
                        final_val = self.get_label_from_cache(val.get('id'))

                    elif dt == "quantity":
                        amount = val.get('amount', '').replace('+', '')
                        unit_url = val.get('unit', '')
                        unit = ""
                        if "entity/" in unit_url:
                            unit = self.get_label_from_cache(unit_url.split('/')[-1])
                        final_val = f"{amount} {unit}".strip()

                    elif dt == "time":
                        final_val = val.get('time', '')
                    elif dt == "monolingualtext":
                        final_val = val.get('text', '')
                    elif dt == "globe-coordinate":
                        lat = val.get('latitude')
                        lon = val.get('longitude')
                        final_val = f"lat:{lat}, lon:{lon}"
                    elif dt == "url" or dt == "commonsMedia":
                        final_val = str(val)
                    else:
                        continue

                    if prop_label in entity_data:
                        entity_data[prop_label] += f"; {final_val}"
                    else:
                        entity_data[prop_label] = final_val

            return entity_data

        except Exception as e:
            print(f"[Wikidata] Process error for {query_name}: {e}")
            return None


if __name__ == "__main__":
    client = SimpleWikidataClient()
    data = client.process_entity("4310 Strömholm")
    print(json.dumps(data, indent=2, ensure_ascii=False))