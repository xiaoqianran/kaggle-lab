"""Two-agent auto conversation loop via Kaggle Model Proxy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from common.proxy import chat_messages

AgentId = Literal["a", "b"]


@dataclass
class Agent:
    id: AgentId
    name: str
    persona: str


@dataclass
class Turn:
    agent_id: AgentId
    agent_name: str
    content: str
    round: int


@dataclass
class DualChatResult:
    topic: str
    model: str
    rounds_done: int
    transcript: list[Turn] = field(default_factory=list)
    agents: dict[str, dict[str, str]] = field(default_factory=dict)


def _build_system(agent: Agent, other: Agent, topic: str) -> str:
    return "\n".join(
        [
            "你正在参与一场自动多轮对话实验。",
            f"你的身份：{agent.name}",
            f"你的人设/立场：{agent.persona or '有主见、善于推进对话的讨论者'}",
            f"对话伙伴：{other.name}（{other.persona or '另一位讨论者'}）",
            f"话题：{topic}",
            "规则：",
            "- 始终保持角色，语气自然，像真人在聊天/辩论",
            "- 必须接住对方上一句，引用或反驳其中一点",
            "- 每次说 2–4 句完整中文，信息密度高；每句必须说完，禁止半截话",
            "- 不要输出 markdown 标题，不要说“作为 AI”",
            "- 不要代替对方说话，不要结束整场对话（除非对方明显收束）",
            "- 语言与话题一致：中文话题用中文",
            "- 控制篇幅：总字数约 80–180 字，宁可短而完整，不要长到被截断",
        ]
    )


def _build_messages(
    *,
    agent: Agent,
    other: Agent,
    topic: str,
    history: list[Turn],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _build_system(agent, other, topic)}
    ]
    for t in history[-16:]:
        role = "assistant" if t.agent_id == agent.id else "user"
        messages.append(
            {"role": role, "content": f"{t.agent_name}: {t.content}"}
        )
    if not history:
        user = (
            f"话题：{topic}\n"
            f"请以 {agent.name} 的身份先开场，直接发言，不要前缀自己的名字。"
        )
    else:
        user = (
            f"继续对话。你是 {agent.name}，请直接回应对方上一句，推进讨论。"
            "不要输出名字前缀，不要总结整场，只说这一轮。"
        )
    messages.append({"role": "user", "content": user})
    return messages


def _strip_name_prefix(text: str, name: str) -> str:
    re_name = re.escape(name)
    return re.sub(rf"^\s*{re_name}\s*[:：\-]\s*", "", text).strip()


def generate_turn(
    *,
    topic: str,
    agent: Agent,
    other: Agent,
    history: list[Turn],
    model: str,
    max_tokens: int = 640,
    temperature: float = 0.9,
    refresh: bool = False,
    experiment: str = "015-dual-agent-chat",
) -> str:
    messages = _build_messages(
        agent=agent, other=other, topic=topic, history=history
    )
    text = chat_messages(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        refresh=refresh,
        experiment=experiment,
    )
    return _strip_name_prefix(text, agent.name)


def run_dual_chat(
    *,
    topic: str,
    agent_a: Agent,
    agent_b: Agent,
    rounds: int = 4,
    model: str,
    max_tokens: int = 640,
    temperature: float = 0.9,
    refresh_first: bool = False,
    on_turn: Callable[[Turn], None] | None = None,
) -> DualChatResult:
    """A speaks then B, repeated `rounds` times (1 round = both speak once)."""
    history: list[Turn] = []
    order: list[Agent] = [agent_a, agent_b]
    first = True

    for r in range(1, max(1, rounds) + 1):
        for agent in order:
            other = agent_b if agent.id == "a" else agent_a
            text = generate_turn(
                topic=topic,
                agent=agent,
                other=other,
                history=history,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                refresh=refresh_first and first,
            )
            first = False
            turn = Turn(
                agent_id=agent.id,
                agent_name=agent.name,
                content=text,
                round=r,
            )
            history.append(turn)
            if on_turn:
                on_turn(turn)

    return DualChatResult(
        topic=topic,
        model=model,
        rounds_done=rounds,
        transcript=history,
        agents={
            "a": {"name": agent_a.name, "persona": agent_a.persona},
            "b": {"name": agent_b.name, "persona": agent_b.persona},
        },
    )


def result_to_dict(res: DualChatResult) -> dict[str, Any]:
    return {
        "topic": res.topic,
        "model": res.model,
        "rounds_done": res.rounds_done,
        "agents": res.agents,
        "transcript": [
            {
                "round": t.round,
                "agent_id": t.agent_id,
                "agent_name": t.agent_name,
                "content": t.content,
            }
            for t in res.transcript
        ],
    }


def result_to_markdown(res: DualChatResult) -> str:
    lines = [
        f"# 双智对谈",
        "",
        f"- **话题**: {res.topic}",
        f"- **模型**: `{res.model}`",
        f"- **轮数**: {res.rounds_done}",
        f"- **A**: {res.agents.get('a', {}).get('name')}",
        f"- **B**: {res.agents.get('b', {}).get('name')}",
        "",
        "---",
        "",
    ]
    for t in res.transcript:
        lines.append(f"### R{t.round} · {t.agent_name}")
        lines.append("")
        lines.append(t.content)
        lines.append("")
    return "\n".join(lines)
