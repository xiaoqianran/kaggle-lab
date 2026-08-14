"""Dual-agent chat presets (topic + two personas)."""

from __future__ import annotations

from typing import Any

# Cheap + good models measured on Kaggle Model Proxy (2026-08)
MODELS: list[dict[str, str]] = [
    {
        "id": "google/gemini-3.5-flash-lite",
        "tag": "推荐",
        "label": "Gemini 3.5 Flash-Lite",
        "blurb": "文笔自然，对谈默认",
    },
    {
        "id": "google/gemini-3.1-flash-lite-preview",
        "tag": "最便宜",
        "label": "Gemini 3.1 Flash-Lite",
        "blurb": "极限省额度，够用",
    },
    {
        "id": "openai/gpt-5.4-nano-2026-03-17",
        "tag": "稳",
        "label": "GPT-5.4 nano",
        "blurb": "中文稳、讲理清楚",
    },
]

DEFAULT_MODEL = MODELS[0]["id"]

PRESETS: dict[str, dict[str, Any]] = {
    "debate": {
        "label": "激辩",
        "topic": "远程办公会不会永久削弱公司创造力？",
        "a": {
            "name": "构建者",
            "persona": "乐观的产品负责人，相信流程与异步协作能放大创造力，喜欢给可执行方案",
        },
        "b": {
            "name": "怀疑者",
            "persona": "犀利的组织行为学者，擅长拆穿空话，强调面对面偶然碰撞与文化密度",
        },
    },
    "socratic": {
        "label": "苏格拉底",
        "topic": "什么叫真正理解一个概念？",
        "a": {
            "name": "提问者",
            "persona": "苏格拉底式导师，少下结论，多用反问逼对方澄清定义与边界",
        },
        "b": {
            "name": "求知者",
            "persona": "认真的研究生，愿意修正观点，试图用例子与反例把抽象说清楚",
        },
    },
    "brainstorm": {
        "label": "脑暴",
        "topic": "给「一个人住的智能公寓」想 5 个反直觉功能",
        "a": {
            "name": "发散机",
            "persona": "疯狂点子机器，跨界联想，不怕离谱，优先数量与惊喜",
        },
        "b": {
            "name": "落地官",
            "persona": "务实工程师，快速评估可行性、成本与用户痛点，把点子剪成 MVP",
        },
    },
    "comedy": {
        "label": "互怼",
        "topic": "为什么程序员总说「明天一定重构」？",
        "a": {
            "name": "老码农",
            "persona": "自嘲型资深工程师，黑色幽默，承认技术债是生活方式",
        },
        "b": {
            "name": "产品经理",
            "persona": "嘴毒但讲理的 PM，用业务压力回怼，擅长把锅抛得优雅",
        },
    },
}
