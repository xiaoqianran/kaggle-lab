import { useCallback, useEffect, useRef, useState } from "react";
import {
  Bot,
  Pause,
  Play,
  RotateCcw,
  Square,
  Settings2,
  MessageSquare,
  Sparkles,
  Zap,
  ExternalLink,
} from "lucide-react";
import {
  DEFAULT_MODEL,
  MODEL_OPTIONS,
  PRESETS,
  type AgentId,
  type AgentProfile,
  type PresetId,
} from "./presets";
import {
  generateTurn,
  loadSettings,
  saveSettings,
  type ChatMessage,
  type StoredSettings,
} from "./dualChat";

type RunState = "idle" | "running" | "paused" | "done";

export default function App() {
  const [topic, setTopic] = useState<string>(PRESETS.debate.topic);
  const [agentA, setAgentA] = useState<AgentProfile>({
    id: "a",
    name: PRESETS.debate.a.name,
    persona: PRESETS.debate.a.persona,
  });
  const [agentB, setAgentB] = useState<AgentProfile>({
    id: "b",
    name: PRESETS.debate.b.name,
    persona: PRESETS.debate.b.persona,
  });
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [maxRounds, setMaxRounds] = useState(6);
  const [delayMs, setDelayMs] = useState(500);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [runState, setRunState] = useState<RunState>("idle");
  const [typing, setTyping] = useState<AgentId | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sourceTag, setSourceTag] = useState<"kaggle" | "demo" | null>(null);
  const [showSetup, setShowSetup] = useState(true);
  const [settings, setSettings] = useState<StoredSettings>(() => loadSettings());
  const [showCreds, setShowCreds] = useState(false);

  const abortRef = useRef(false);
  const pauseRef = useRef(false);
  const loopIdRef = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<ChatMessage[]>([]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, typing]);
  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  const applyPreset = (id: PresetId) => {
    const p = PRESETS[id];
    setTopic(p.topic);
    setAgentA({ id: "a", name: p.a.name, persona: p.a.persona });
    setAgentB({ id: "b", name: p.b.name, persona: p.b.persona });
  };

  const isBusy = runState === "running" || runState === "paused";

  const stopLoop = () => {
    abortRef.current = true;
    pauseRef.current = false;
    loopIdRef.current += 1;
    setTyping(null);
    setRunState((s) => (s === "running" || s === "paused" ? "idle" : s));
  };

  const runConversation = useCallback(async () => {
    if (!topic.trim()) {
      setError("先写一个话题");
      return;
    }
    if (settings.mode === "live" && !settings.apiBase.trim()) {
      setError(
        "Live 需要 API Base：Cloudflare Worker 的 https://xxx.workers.dev/api/openai，或本机 http://127.0.0.1:8765/api/openai",
      );
      setShowCreds(true);
      return;
    }
    // apiKey 可留空：CF Worker / 本机 gateway 用服务端 Secret 自动刷 token

    abortRef.current = false;
    pauseRef.current = false;
    const myLoop = ++loopIdRef.current;
    setError(null);
    setMessages([]);
    messagesRef.current = [];
    setRunState("running");
    setShowSetup(false);

    const rounds = Math.min(Math.max(maxRounds, 1), 20);
    let next: AgentId = "a";

    for (let r = 1; r <= rounds; r++) {
      for (let turn = 0; turn < 2; turn++) {
        if (abortRef.current || myLoop !== loopIdRef.current) return;
        while (pauseRef.current && !abortRef.current) {
          await new Promise((res) => setTimeout(res, 120));
        }
        if (abortRef.current || myLoop !== loopIdRef.current) return;

        const agent = next === "a" ? agentA : agentB;
        const other = next === "a" ? agentB : agentA;
        setTyping(next);
        try {
          const result = await generateTurn(
            {
              apiBase: settings.apiBase,
              apiKey: settings.apiKey,
              model: modelId,
              maxTokens: 640,
              mode: settings.mode,
            },
            topic.trim(),
            agent,
            other,
            messagesRef.current,
          );
          if (abortRef.current || myLoop !== loopIdRef.current) return;
          setSourceTag(result.source);
          const msg: ChatMessage = {
            id: `${Date.now()}-${next}-${r}`,
            agentId: next,
            agentName: agent.name,
            content: result.text,
            round: r,
            ts: Date.now(),
          };
          setMessages((prev) => {
            const n = [...prev, msg];
            messagesRef.current = n;
            return n;
          });
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          const corsHint =
            /Failed to fetch|NetworkError|CORS/i.test(msg)
              ? " — 浏览器直连 Model Proxy 通常被 CORS 拦截。请用仓库 gateway.py 作中转，或把 API Base 指到你的网关 /api/openai。"
              : "";
          setError(msg + corsHint);
          setTyping(null);
          setRunState("idle");
          return;
        }
        setTyping(null);
        next = next === "a" ? "b" : "a";
        if (delayMs > 0) await new Promise((res) => setTimeout(res, delayMs));
      }
    }
    if (!abortRef.current && myLoop === loopIdRef.current) setRunState("done");
  }, [topic, agentA, agentB, maxRounds, delayMs, modelId, settings]);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">
            <Bot size={20} />
          </div>
          <div>
            <h1>双智对谈</h1>
            <p>kaggle-lab · 015 · Kaggle Model Proxy</p>
          </div>
        </div>
        <div className="badges">
          <span className={`badge ${settings.mode === "live" ? "live" : "warn"}`}>
            {settings.mode === "live" ? "Live Proxy" : "演示模式"}
          </span>
          {sourceTag === "kaggle" && <span className="badge live">本轮 Kaggle</span>}
          {sourceTag === "demo" && runState !== "idle" && <span className="badge">本轮演示</span>}
          <span className="badge">{modelId.split("/").pop()}</span>
          <a
            className="badge"
            href="https://github.com/xiaoqianran/kaggle-lab"
            target="_blank"
            rel="noreferrer"
            style={{ textDecoration: "none" }}
          >
            GitHub <ExternalLink size={12} style={{ marginLeft: 4 }} />
          </a>
        </div>
      </header>

      <div className="layout">
        <aside className={`panel ${showSetup ? "" : "setup-collapsed"}`}>
          <div className="panel-title">
            <span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
              <Settings2 size={16} /> 配置
            </span>
            <button type="button" className="chip mobile-only" onClick={() => setShowSetup(false)}>
              收起
            </button>
          </div>

          <div className="field">
            <label>运行模式</label>
            <div className="chips">
              <button
                type="button"
                className={`chip ${settings.mode === "demo" ? "active" : ""}`}
                disabled={isBusy}
                onClick={() => setSettings((s) => ({ ...s, mode: "demo" }))}
              >
                演示（无需 key）
              </button>
              <button
                type="button"
                className={`chip ${settings.mode === "live" ? "active" : ""}`}
                disabled={isBusy}
                onClick={() => {
                  setSettings((s) => ({ ...s, mode: "live" }));
                  setShowCreds(true);
                }}
              >
                Live（Model Proxy）
              </button>
            </div>
          </div>

          {(showCreds || settings.mode === "live") && (
            <div className="field">
              <label>凭证（仅存本机浏览器 localStorage，勿提交仓库）</label>
              <p className="hint" style={{ marginBottom: 8 }}>
                Pages Live 推荐填 CF Worker：
                <code>https://你的worker.workers.dev/api/openai</code>
                ；Key 可留空。本机网关：
                <code>http://127.0.0.1:8765/api/openai</code>
              </p>
              <input
                placeholder="API Base：https://…/openapi 或 https://你的网关/api/openai"
                disabled={isBusy}
                value={settings.apiBase}
                onChange={(e) => setSettings((s) => ({ ...s, apiBase: e.target.value }))}
              />
              <input
                type="password"
                placeholder="MODEL_PROXY_API_KEY（kaggle b auth）"
                disabled={isBusy}
                value={settings.apiKey}
                onChange={(e) => setSettings((s) => ({ ...s, apiKey: e.target.value }))}
                style={{ marginTop: 6 }}
              />
              <span className="hint">
                Pages 无法安全保管密钥。本地：<code>python gateway.py</code> 后 API Base 用{" "}
                <code>http://127.0.0.1:8765/api/openai</code>
              </span>
            </div>
          )}

          <div className="chips" style={{ marginBottom: 12 }}>
            {(Object.keys(PRESETS) as PresetId[]).map((id) => (
              <button
                key={id}
                type="button"
                className="chip"
                disabled={isBusy}
                onClick={() => applyPreset(id)}
              >
                {PRESETS[id].label}
              </button>
            ))}
          </div>

          <div className="field">
            <label>话题</label>
            <textarea disabled={isBusy} value={topic} onChange={(e) => setTopic(e.target.value)} />
          </div>

          <AgentCard
            accent="a"
            label="智能体 A"
            agent={agentA}
            disabled={isBusy}
            onChange={setAgentA}
          />
          <AgentCard
            accent="b"
            label="智能体 B"
            agent={agentB}
            disabled={isBusy}
            onChange={setAgentB}
          />

          <div className="field">
            <label>模型</label>
            <select
              disabled={isBusy}
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.tag} · {m.label}
                </option>
              ))}
            </select>
            <span className="hint">{MODEL_OPTIONS.find((m) => m.id === modelId)?.blurb}</span>
            <div className="chips" style={{ marginTop: 6 }}>
              {MODEL_OPTIONS.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  className={`chip ${modelId === m.id ? "active" : ""}`}
                  disabled={isBusy}
                  onClick={() => setModelId(m.id)}
                >
                  {m.tag}
                </button>
              ))}
            </div>
          </div>

          <div className="grid-2">
            <div className="field">
              <label>最大轮数</label>
              <input
                type="number"
                min={1}
                max={20}
                disabled={isBusy}
                value={maxRounds}
                onChange={(e) => setMaxRounds(Number(e.target.value) || 1)}
              />
            </div>
            <div className="field">
              <label>间隔 ms</label>
              <input
                type="number"
                min={0}
                max={5000}
                step={100}
                disabled={isBusy}
                value={delayMs}
                onChange={(e) => setDelayMs(Number(e.target.value) || 0)}
              />
            </div>
          </div>

          <div className="btn-row">
            {runState === "idle" || runState === "done" ? (
              <button type="button" className="btn primary" onClick={() => void runConversation()}>
                <Play size={16} /> 开始自动对谈
              </button>
            ) : runState === "running" ? (
              <div className="btn-row two">
                <button
                  type="button"
                  className="btn secondary"
                  onClick={() => {
                    pauseRef.current = true;
                    setRunState("paused");
                  }}
                >
                  <Pause size={16} /> 暂停
                </button>
                <button type="button" className="btn danger" onClick={stopLoop}>
                  <Square size={16} /> 停止
                </button>
              </div>
            ) : (
              <div className="btn-row two">
                <button
                  type="button"
                  className="btn success"
                  onClick={() => {
                    pauseRef.current = false;
                    setRunState("running");
                  }}
                >
                  <Play size={16} /> 继续
                </button>
                <button type="button" className="btn danger" onClick={stopLoop}>
                  <Square size={16} /> 停止
                </button>
              </div>
            )}
            <button
              type="button"
              className="btn ghost"
              disabled={isBusy || messages.length === 0}
              onClick={() => {
                setMessages([]);
                setRunState("idle");
                setError(null);
                setSourceTag(null);
              }}
            >
              <RotateCcw size={16} /> 清空记录
            </button>
          </div>
          {error && <div className="error">{error}</div>}
        </aside>

        <section className="panel stage">
          <div className="stage-head">
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <MessageSquare size={16} color="var(--fg-muted)" />
              <strong style={{ fontSize: 14 }}>对谈现场</strong>
              <StatusBadge state={runState} />
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12, color: "var(--fg-subtle)" }}>
              <Zap size={14} />
              {messages.length} 条 · 第 {Math.max(1, Math.ceil(messages.length / 2))} / {maxRounds} 轮
              <button type="button" className="chip mobile-only" onClick={() => setShowSetup(true)}>
                配置
              </button>
            </div>
          </div>
          <div className="stage-agents">
            <div className="a">
              <span className="dot" />
              {agentA.name}
            </div>
            <div className="b">
              <span className="dot" />
              {agentB.name}
            </div>
          </div>
          <div className="feed">
            {messages.length === 0 && !typing && (
              <div className="empty">
                <div>
                  <div
                    style={{
                      width: 56,
                      height: 56,
                      margin: "0 auto 12px",
                      borderRadius: 999,
                      border: "1px solid var(--border)",
                      display: "grid",
                      placeItems: "center",
                      background: "var(--bg-subtle)",
                    }}
                  >
                    <Sparkles size={22} />
                  </div>
                  <h2>还没开聊</h2>
                  <p>
                    <span style={{ color: "var(--agent-a)" }}>{agentA.name}</span>
                    {" 与 "}
                    <span style={{ color: "var(--agent-b)" }}>{agentB.name}</span>
                    {" 将围绕「"}
                    <span style={{ color: "var(--fg)" }}>{topic}</span>
                    {"」自动多轮对话。"}
                  </p>
                  <button
                    type="button"
                    className="btn primary"
                    style={{ marginTop: 16 }}
                    onClick={() => void runConversation()}
                  >
                    <Play size={16} /> 开始自动对谈
                  </button>
                </div>
              </div>
            )}
            {messages.map((m) => (
              <div key={m.id} className={`msg-enter bubble-wrap ${m.agentId}`}>
                <div className={`bubble-meta ${m.agentId}`}>
                  {m.agentName}
                  <span className="r">R{m.round}</span>
                </div>
                <div className={`bubble ${m.agentId}`}>{m.content}</div>
              </div>
            ))}
            {typing && (
              <div className={`msg-enter bubble-wrap ${typing}`}>
                <div className={`bubble-meta ${typing}`}>
                  {typing === "a" ? agentA.name : agentB.name} 正在组织语言
                </div>
                <div className={`bubble ${typing} typing-row`}>
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
          {runState === "done" && (
            <div style={{ borderTop: "1px solid var(--border)", padding: 12, textAlign: "center", fontSize: 14, color: "var(--fg-muted)" }}>
              本场结束。
              <button type="button" className="btn primary" style={{ marginLeft: 12, height: 32 }} onClick={() => void runConversation()}>
                再来一轮
              </button>
            </div>
          )}
        </section>
      </div>

      <p className="footer-note">
        实验目录 <code>labs/015-dual-agent-chat</code> · CLI:{" "}
        <code>python -m kaggle_lab debate</code> ·{" "}
        <a href="https://github.com/xiaoqianran/kaggle-lab/tree/main/labs/015-dual-agent-chat">源码</a>
      </p>
    </div>
  );
}

function AgentCard({
  accent,
  label,
  agent,
  disabled,
  onChange,
}: {
  accent: "a" | "b";
  label: string;
  agent: AgentProfile;
  disabled?: boolean;
  onChange: (a: AgentProfile) => void;
}) {
  return (
    <div className={`agent-card ${accent}`}>
      <span className={`badge ${accent}`}>{label}</span>
      <div className="field" style={{ marginTop: 8, marginBottom: 8 }}>
        <label>名字</label>
        <input
          disabled={disabled}
          value={agent.name}
          onChange={(e) => onChange({ ...agent, name: e.target.value })}
        />
      </div>
      <div className="field" style={{ marginBottom: 0 }}>
        <label>人设</label>
        <textarea
          disabled={disabled}
          value={agent.persona}
          onChange={(e) => onChange({ ...agent, persona: e.target.value })}
          style={{ minHeight: 64, fontSize: 12 }}
        />
      </div>
    </div>
  );
}

function StatusBadge({ state }: { state: RunState }) {
  if (state === "running") return <span className="badge live">对谈中</span>;
  if (state === "paused") return <span className="badge warn">已暂停</span>;
  if (state === "done") return <span className="badge">已结束</span>;
  return <span className="badge">待命</span>;
}
