"""Query-aware source selection and citation-safe synthesis inputs."""
from __future__ import annotations

import re
from urllib.parse import urlsplit


def relevance(query: str, source: dict) -> float:
    text = " ".join(str(source.get(k, "")) for k in ("title", "source_span", "snippet")).lower()
    host = urlsplit(str(source.get("url", ""))).netloc.lower()
    us_query = bool(re.search(r"美股|美国|美联储|\bu\.?s\.?\b|united states", query, re.I))
    if us_query:
        us_terms = r"美国|美股|美联储|标普|纳斯达克|道琼斯|非农|federal reserve|united states|s&p|nasdaq|dow jones|\bu\.?s\.?\b"
        if not re.search(us_terms, text):
            if host.endswith((".gov.cn", ".gov.hk")) or re.search(r"香港|无锡|中国居民|hong kong", text):
                return -1.0
        if re.search(us_terms, text):
            country_score = 2.0
        else:
            country_score = 0.0
    else:
        country_score = 0.0
    tokens = re.findall(r"[a-z]{3,}|[\u4e00-\u9fff]+", query.lower())
    terms = set()
    for token in tokens:
        if re.search(r"[\u4e00-\u9fff]", token):
            terms.update(token[i:i+2] for i in range(len(token)-1))
        else:
            terms.add(token)
    terms -= {"原因", "分析", "最近", "一个", "个月", "什么", "the", "and", "why"}
    return country_score + sum(term in text for term in terms) / max(len(terms), 1)


def remap_citations(content: str, old_sources: list[dict], new_sources: list[dict]) -> str:
    """Never let an old numeric reference silently point to a new source."""
    old = {str(s.get("citation_id", i)): s for i, s in enumerate(old_sources, 1)}
    new = {str(s.get("source_id") or s.get("url")): s["citation_id"] for s in new_sources}

    def replace(match):
        source = old.get(match.group(1), {})
        identity = str(source.get("source_id") or source.get("url"))
        citation = new.get(identity)
        return f"[{citation}]" if citation is not None else "（原引用不在当前目录，需重新核验）"

    return re.sub(r"\[(\d+)\]", replace, content)
