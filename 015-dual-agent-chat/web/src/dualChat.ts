import type { AgentId, AgentProfile } from "./presets";
import { simulateTurn } from "./simulate";

export type ChatMessage = {
  id: string;
  agentId: AgentId;
  agentName: string;
  content: string;
  round: number;
  ts: number;
};

export type RuntimeConfig = {
  /** OpenAI-compatible base, e.g. https://mp.../models/openapi OR gateway /api/openai */
  apiBase: string;
  apiKey: string;
  model: string;
  maxTokens: number;
  mode: "live" | "demo";
};

function buildSystem(agent: AgentProfile, other: AgentProfile, topic: string): string {
  return [
    "你正在参与一场自动多轮对话实验。",
    `你的身份：${agent.name}`,
    `你的人设/立场：${agent.persona || "有主见、善于推进对话的讨论者"}`,
    `对话伙伴：${other.name}（${other.persona || "另一位讨论者"}）`,
    `话题：${topic}`,
    "规则：",
    "- 始终保持角色，语气自然，像真人在聊天/辩论",
    "- 必须接住对方上一句，引用或反驳其中一点",
    "- 每次说 2–4 句完整中文，信息密度高；每句必须说完，禁止半截话",
    "- 不要输出 markdown 标题，不要说“作为 AI”",
    "- 不要代替对方说话，不要结束整场对话（除非对方明显收束）",
    "- 语言与话题一致：中文话题用中文",
    "- 控制篇幅：总字数约 80–180 字，宁可短而完整，不要长到被截断",
  ].join("\n");
}

function buildMessages(
  topic: string,
  agent: AgentProfile,
  other: AgentProfile,
  history: ChatMessage[],
) {
  const messages: { role: "system" | "user" | "assistant"; content: string }[] = [
    { role: "system", content: buildSystem(agent, other, topic) },
  ];
  for (const m of history.slice(-16)) {
    messages.push({
      role: m.agentId === agent.id ? "assistant" : "user",
      content: `${m.agentName}: ${m.content}`,
    });
  }
  messages.push({
    role: "user",
    content:
      history.length === 0
        ? `话题：${topic}\n请以 ${agent.name} 的身份先开场，直接发言，不要前缀自己的名字。`
        : `继续对话。你是 ${agent.name}，请直接回应对方上一句，推进讨论。不要输出名字前缀，不要总结整场，只说这一轮。`,
  });
  return messages;
}

function stripNamePrefix(text: string, name: string): string {
  const re = new RegExp(`^\\s*${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*[:：\\-]\\s*`, "i");
  return text.replace(re, "").trim();
}

export async function generateTurn(
  cfg: RuntimeConfig,
  topic: string,
  agent: AgentProfile,
  other: AgentProfile,
  history: ChatMessage[],
): Promise<{ text: string; source: "kaggle" | "demo" }> {
  if (cfg.mode === "demo" || !cfg.apiKey.trim()) {
    await sleep(280 + Math.random() * 320);
    return {
      text: simulateTurn({
        topic,
        agent,
        other,
        history: history.map((h) => ({
          agentId: h.agentId,
          agentName: h.agentName,
          content: h.content,
        })),
      }),
      source: "demo",
    };
  }

  const base = cfg.apiBase.replace(/\/$/, "");
  const url = `${base}/chat/completions`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${cfg.apiKey.trim()}`,
      },
      body: JSON.stringify({
        model: cfg.model,
        messages: buildMessages(topic, agent, other, history),
        max_tokens: cfg.maxTokens,
        temperature: 0.9,
      }),
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(
      `网络/CORS 失败（${msg}）。GitHub Pages 不能直连 Kaggle Proxy：` +
        `请用本地网关 python 015-dual-agent-chat/gateway.py，` +
        `API Base 填 http://127.0.0.1:8765/api/openai（仅本机）或你的公网网关。` +
        `纯演示请切换 Demo 模式。`,
    );
  }

  if (!res.ok) {
    const err = await res.text().catch(() => "");
    throw new Error(`Proxy HTTP ${res.status}: ${err.slice(0, 240)}`);
  }
  const body = (await res.json()) as {
    choices?: { message?: { content?: string | null } }[];
  };
  const text = body.choices?.[0]?.message?.content?.trim();
  if (!text) throw new Error("模型返回空内容");
  return { text: stripNamePrefix(text, agent.name), source: "kaggle" };
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

const LS_KEY = "kaggle-lab-015-settings-v1";

export type StoredSettings = {
  apiBase: string;
  apiKey: string;
  mode: "live" | "demo";
};

export function loadSettings(): StoredSettings {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return defaultSettings();
    return { ...defaultSettings(), ...JSON.parse(raw) };
  } catch {
    return defaultSettings();
  }
}

export function saveSettings(s: StoredSettings) {
  localStorage.setItem(LS_KEY, JSON.stringify(s));
}

export function defaultSettings(): StoredSettings {
  // Same-origin gateway when served by gateway.py; on Pages leave blank until user fills.
  const isPages =
    typeof location !== "undefined" && location.hostname.endsWith("github.io");
  return {
    apiBase: isPages ? "" : `${location.origin}/api/openai`,
    apiKey: "",
    mode: isPages ? "demo" : "live",
  };
}
