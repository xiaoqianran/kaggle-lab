"""Declarative map of labs and tracks.

Adding a lab
------------
1. Create ``labs/NNN-topic/run.py`` (argparse subcommands).
2. Append a :class:`Lab` below (id, aliases, title, journey).
3. ``python -m kaggle_lab list`` picks it up.

Filesystem scan is the fallback: an unregistered ``run.py`` still appears
in ``list`` so a new folder is never invisible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from kaggle_lab.paths import LABS_DIR, TRACKS_DIR


@dataclass(frozen=True)
class Lab:
    id: str
    title: str
    summary: str
    group: str
    aliases: tuple[str, ...] = ()
    default_cmd: str = "run"
    uses_proxy: bool = True
    dangerous: bool = False
    danger_note: str = ""
    product: bool = False

    @property
    def dir(self) -> Path:
        return LABS_DIR / self.id

    @property
    def run_py(self) -> Path:
        return self.dir / "run.py"


@dataclass(frozen=True)
class Track:
    id: str
    title: str
    summary: str
    aliases: tuple[str, ...] = ()
    kaggle_users: tuple[str, ...] = ()
    start_here: str = "README.md"

    @property
    def dir(self) -> Path:
        return TRACKS_DIR / self.id


@dataclass(frozen=True)
class Journey:
    """User-intent entry. Commands in CLI are named after these ids."""

    id: str
    title: str
    when: str
    lab_id: str | None = None
    cmd: str | None = None
    extra_hint: str = ""
    # always: journey name IS the lab subcommand (chat "你好")
    # default: inject cmd only when argv is empty or starts with a flag
    inject: str = "default"


# --- labs (learning order, not dump order) ---------------------------------

LABS: tuple[Lab, ...] = (
    Lab(
        id="001-model-proxy",
        title="Model Proxy 聊天",
        summary="刷新临时凭证，发一条 chat。所有扣 AI $ 的实验从这里通。",
        group="start",
        aliases=("001", "proxy"),
        default_cmd="auth",
    ),
    Lab(
        id="002-tool-call",
        title="单轮 Tool Calling",
        summary="模型只返回 tool_calls，本地执行后再回填。",
        group="proxy",
        aliases=("002", "tool-call", "tools"),
    ),
    Lab(
        id="003-list-models",
        title="模型表",
        summary="列出 / 导出 Kaggle Benchmarks 可用模型（不扣额度）。",
        group="start",
        aliases=("003", "models", "list-models"),
        default_cmd="list",
        uses_proxy=False,
    ),
    Lab(
        id="004-sae",
        title="SAE 正式客户端",
        summary="标准化考试 HTTP 客户端。默认禁止 register/start/submit。",
        group="sae",
        aliases=("004", "sae"),
        default_cmd="whoami",
        uses_proxy=False,
        dangerous=True,
        danger_note="默认不开考。只要 dry-run 请用 009。",
    ),
    Lab(
        id="005-benchmark-task",
        title="Benchmark 最小闭环",
        summary="写 task → 本地校验 → push → run。扣 AI Models $。",
        group="platform",
        aliases=("005", "bench", "benchmark"),
        default_cmd="validate",
    ),
    Lab(
        id="006-judge-llm",
        title="多模型裁判",
        summary="同一题库上规则分 + Judge LLM，出排行榜。",
        group="proxy",
        aliases=("006", "judge"),
    ),
    Lab(
        id="007-mcp-harness",
        title="Kaggle MCP",
        summary="官方 MCP 薄客户端（~70 tools）。不扣 AI $。",
        group="platform",
        aliases=("007", "mcp"),
        default_cmd="tools",
        uses_proxy=False,
    ),
    Lab(
        id="008-agent-loop",
        title="手写 ReAct",
        summary="多轮 tool-calling；可选挂上 MCP 搜竞赛。",
        group="proxy",
        aliases=("008", "react", "agent-loop"),
    ),
    Lab(
        id="009-sae-better",
        title="SAE dry-run",
        summary="假卷 + 清洗 + 可选 ensemble。不占正式考试次数。",
        group="sae",
        aliases=("009", "sae-better", "dry-run"),
        default_cmd="dry-run",
        danger_note="只建议 dry-run / show，不要联动 004 交卷。",
    ),
    Lab(
        id="010-quota-dashboard",
        title="额度看板",
        summary="GPU/TPU 周配额 + 本机 AI Models 用量账本。",
        group="start",
        aliases=("010", "quota"),
        default_cmd="show",
        uses_proxy=False,
    ),
    Lab(
        id="011-camel-proxy",
        title="CAMEL 底座",
        summary="CAMEL ChatAgent 接到 Model Proxy。后续 multi-agent 的依赖。",
        group="camel",
        aliases=("011", "camel", "camel-proxy"),
        default_cmd="smoke",
    ),
    Lab(
        id="012-camel-roleplay",
        title="CAMEL RolePlaying",
        summary="双角色任务驱动协作（Lab Lead ↔ Researcher）。",
        group="camel",
        aliases=("012", "roleplay"),
    ),
    Lab(
        id="013-camel-kaggle-crew",
        title="竞赛侦察简报",
        summary="CAMEL + 真 Kaggle 信号（MCP / quota）写出行动简报。",
        group="camel",
        aliases=("013", "crew"),
        default_cmd="scout",
    ),
    Lab(
        id="014-camel-workforce-bench",
        title="Workforce 写 Benchmark",
        summary="侦察→辩论→出题→本地校验→可选 push。publish 必须 --i-accept。",
        group="camel",
        aliases=("014", "workforce"),
    ),
    Lab(
        id="015-dual-agent-chat",
        title="双智对谈",
        summary="自由多轮 A↔B（CLI + Web + 本地网关）。产品入口。",
        group="product",
        aliases=("015", "dual", "dual-chat"),
        product=True,
    ),
)

# --- tracks ----------------------------------------------------------------

TRACKS: tuple[Track, ...] = (
    Track(
        id="instruct-t4",
        title="Mini-Instruct T4",
        summary="LoRA → SFT → DPO → merge；另含 img3d 与 world models。",
        aliases=("v01", "instruct", "lora", "t4"),
        kaggle_users=("wangran521", "seachenbgdy"),
        start_here="README.md",
    ),
    Track(
        id="depth-estimation",
        title="深度估计",
        summary="From-scratch FS00–FS14：几何、光度、foundation、闭环研究。",
        aliases=("v02", "depth", "depth-est"),
        kaggle_users=("shuhuaqaq",),
        start_here="README.md",
    ),
    Track(
        id="image-classification",
        title="图像分类",
        summary="像素→kNN→CNN→现代结构→SSL；P0–P7 与 from-scratch 阶梯。",
        aliases=("v03", "cls", "classification"),
        kaggle_users=("zhengyingxionger",),
        start_here="LEARNING_ROADMAP.md",
    ),
    Track(
        id="object-detection",
        title="目标检测",
        summary="框、两阶段、YOLO、DETR、配方与假设检验；FS00–FS15。",
        aliases=("v04", "det", "detection"),
        kaggle_users=("xiaoshuhuaer",),
        start_here="LEARNING_ROADMAP.md",
    ),
    Track(
        id="diffusion-gemma",
        title="DiffusionGemma T4×2",
        summary="官方 26B-A4B-it 在 Kaggle 双 T4 上跑通（FP16 切卡 + offload）。",
        aliases=("v05", "diffusion", "gemma"),
        kaggle_users=("yaoyunqqq",),
    ),
    Track(
        id="arc-agi-3",
        title="ARC-AGI-3",
        summary="ARC Prize 2026 交卷轨。换环境先读 HANDOFF.md。公开榜 0.17。",
        aliases=("v06", "arc", "arc-agi"),
        kaggle_users=("chutianqiu",),
        start_here="HANDOFF.md",
    ),
)

# --- user journeys ---------------------------------------------------------

JOURNEYS: tuple[Journey, ...] = (
    Journey(
        id="auth",
        title="先接通 Model Proxy",
        when="第一次跑、或临时 key 过期（约 1 小时）",
        lab_id="001-model-proxy",
        cmd="auth",
        inject="always",
    ),
    Journey(
        id="chat",
        title="跟模型聊一句",
        when="确认额度 / 模型是否可用",
        lab_id="001-model-proxy",
        cmd="chat",
        extra_hint='python -m kaggle_lab chat "你好"',
        inject="always",
    ),
    Journey(
        id="debate",
        title="看两个智能体对谈",
        when="想看自由辩论，而不是任务协作",
        lab_id="015-dual-agent-chat",
        cmd="run",
        extra_hint="Web: python -m kaggle_lab gateway",
    ),
    Journey(
        id="workforce",
        title="多角色写一道 Benchmark",
        when="想把额度花在「侦察→辩论→出题」而不是空聊",
        lab_id="014-camel-workforce-bench",
        cmd="run",
    ),
    Journey(
        id="scout",
        title="侦察一场竞赛",
        when="需要带真实 Kaggle 信号的简报",
        lab_id="013-camel-kaggle-crew",
        cmd="scout",
    ),
    Journey(
        id="quota",
        title="看 GPU / AI 额度",
        when="担心烧完 Daily $10 或 GPU 周配额",
        lab_id="010-quota-dashboard",
        cmd="show",
    ),
    Journey(
        id="learn",
        title="跟一条学习轨",
        when="分类 / 检测 / 深度 / Instruct / ARC",
        extra_hint="python -m kaggle_lab track",
    ),
)

GROUP_ORDER = ("start", "proxy", "platform", "camel", "product", "sae")
GROUP_TITLE = {
    "start": "先跑这些",
    "proxy": "Proxy / Agent",
    "platform": "Kaggle 平台",
    "camel": "CAMEL 多智",
    "product": "产品",
    "sae": "SAE（默认不开考）",
}


def labs() -> tuple[Lab, ...]:
    return LABS


def tracks() -> tuple[Track, ...]:
    return TRACKS


def journeys() -> tuple[Journey, ...]:
    return JOURNEYS


def _norm(token: str) -> str:
    return token.strip().lower().rstrip("/")


def resolve_lab(token: str) -> Lab | None:
    """Exact id, unique alias, or unique numeric/prefix match."""
    key = _norm(token)
    if not key:
        return None
    by_id = {lab.id: lab for lab in LABS}
    if key in by_id:
        return by_id[key]
    alias_hits = [lab for lab in LABS if key in lab.aliases]
    if len(alias_hits) == 1:
        return alias_hits[0]
    if len(alias_hits) > 1:
        names = ", ".join(l.id for l in alias_hits)
        raise SystemExit(f"ambiguous lab '{token}': {names} — use full id")
    prefix = [lab for lab in LABS if lab.id == key or lab.id.startswith(key + "-")]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        names = ", ".join(l.id for l in prefix)
        raise SystemExit(f"ambiguous lab '{token}': {names} — use full id")
    return None


def resolve_track(token: str) -> Track | None:
    key = _norm(token)
    if not key:
        return None
    for tr in TRACKS:
        if key == tr.id or key in tr.aliases:
            return tr
    prefix = [tr for tr in TRACKS if tr.id.startswith(key)]
    if len(prefix) == 1:
        return prefix[0]
    return None


def discover_unregistered_labs() -> list[Path]:
    """Folders with run.py that are not in the catalog (extensibility)."""
    if not LABS_DIR.is_dir():
        return []
    known = {lab.id for lab in LABS}
    extra: list[Path] = []
    for p in sorted(LABS_DIR.iterdir()):
        if p.name.startswith(".") or p.name.startswith("_"):
            continue
        if p.is_dir() and (p / "run.py").is_file() and p.name not in known:
            extra.append(p)
    return extra


def discover_unregistered_tracks() -> list[Path]:
    if not TRACKS_DIR.is_dir():
        return []
    known = {tr.id for tr in TRACKS}
    extra: list[Path] = []
    for p in sorted(TRACKS_DIR.iterdir()):
        if p.name.startswith(".") or p.name.startswith("_"):
            continue
        if p.is_dir() and p.name not in known:
            extra.append(p)
    return extra


def lab_by_id(lab_id: str) -> Lab:
    lab = resolve_lab(lab_id)
    if lab is None:
        raise KeyError(lab_id)
    return lab


def iter_labs_by_group() -> Iterable[tuple[str, list[Lab]]]:
    buckets: dict[str, list[Lab]] = {g: [] for g in GROUP_ORDER}
    other: list[Lab] = []
    for lab in LABS:
        if lab.group in buckets:
            buckets[lab.group].append(lab)
        else:
            other.append(lab)
    for g in GROUP_ORDER:
        if buckets[g]:
            yield g, buckets[g]
    if other:
        yield "other", other
