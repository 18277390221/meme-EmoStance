#!/usr/bin/env python3
"""文档场景的可选规则建议；不是语言模型，不改原始转写或执行指令。"""
import argparse
from dataclasses import dataclass
import json
import re


@dataclass(frozen=True)
class Review:
    raw: str
    suggestion: str
    changes: tuple


# 这是显式的候选词表，不是用户实测错误语料。所有改动均须确认。
EDIT_ALIAS = r'(?:边际|边集|编集|编缉)'
DEDUPE_ALIAS = r'(?:去虫|去从|去冲|去充|取重)'
DOCUMENT = r'(?:PDF|文档|文件|文字|文本|扫描件)'
IMAGE = r'(?:图片|图像|照片|图档)'
EDITABLE = re.compile(r'可(?P<word>' + EDIT_ALIAS + r')(?=\s*(?:的\s*)?' + DOCUMENT + r')', re.I)
EDIT_ACTION = re.compile(r'(?P<word>' + EDIT_ALIAS + r')(?=\s*(?:(?:一下|这份|这个|该|我的|这张)\s*)?' + DOCUMENT + r')', re.I)
DEDUPE = re.compile(r'(?P<word>' + DEDUPE_ALIAS + r')(?=\s*(?:一下|这些|这批|重复的)?\s*' + IMAGE + r')', re.I)


def review_documents(text):
    """只提出有局部搭配约束的候选；不输出置信度或猜测通用语义。"""
    replacements = []
    for match in EDITABLE.finditer(text):
        replacements.append((*match.span('word'), '编辑', '可…文档/PDF搭配'))
    for match in EDIT_ACTION.finditer(text):
        before = text[max(0, match.start()-8):match.start()]
        if re.search(r'(?:请|帮我|帮忙|重新|需要|进行|打开后|可以)\s*$', before):
            replacements.append((*match.span('word'), '编辑', '文档编辑动作搭配'))
    # “去虫”本身可以有合法含义，仅在图片+文档制作/重复处理语境下提示，仍不自动采用。
    workflow = re.search(r'(?:制作|生成|做成|转换|转成|合成).{0,12}(?:PDF|文档)', text, re.I)
    duplicates = re.search(r'重复(?:的|图片|照片|图像|内容)|删(?:掉|除)重复', text)
    if workflow or duplicates:
        for match in DEDUPE.finditer(text):
            replacements.append((*match.span('word'), '去重', '图片去重与文档处理语境'))
    # 合并重复命中且保留所有未命中的原文，包括空格、大小写和标点。
    changes, seen, last = [], set(), 0
    pieces = []
    for start, end, word, reason in sorted(replacements):
        if (start, end) in seen or start < last:
            continue
        seen.add((start, end))
        pieces.extend((text[last:start], word))
        changes.append({'original': text[start:end], 'candidate': word, 'reason': reason})
        last = end
    pieces.append(text[last:])
    return Review(text, ''.join(pieces), tuple(changes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True, help='已有识别文字；不录音、不联网')
    args = parser.parse_args()
    result = review_documents(args.text)
    print(json.dumps({'raw_text': result.raw, 'suggested_text': result.suggestion,
                      'changes': result.changes, 'requires_confirmation': bool(result.changes)}, ensure_ascii=False))
