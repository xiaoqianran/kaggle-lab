import type { AgentId, AgentProfile } from "./presets";

type Hist = { agentId: AgentId; agentName: string; content: string };

/** Offline demo when no Model Proxy key / CORS blocked. */
export function simulateTurn(input: {
  topic: string;
  agent: AgentProfile;
  other: AgentProfile;
  history: Hist[];
}): string {
  const { topic, agent, other, history } = input;
  const n = history.length;
  const last = history[history.length - 1];
  const lastText = last?.content ?? "";
  const persona = agent.persona.replace(/\s+/g, " ").trim().slice(0, 36) || "讨论者";
  if (n === 0) {
    return agent.id === "a"
      ? `我是${agent.name}。围绕「${topic}」，关键不在口号，而在可验证的机制。从${persona}的视角，先拆成目标、约束与可执行步骤。你怎么看？`
      : `我是${agent.name}。我不买账把「${topic}」说得太顺。从${persona}出发，先质疑前提：我们是在解决问题，还是换一种说法？`;
  }
  const hook =
    lastText
      .split(/[，。！？、；：\n,.!?;:]/)
      .map((s) => s.trim())
      .filter((s) => s.length >= 4 && s.length <= 24)[0] || topic;
  const round = Math.floor(n / 2) + 1;
  if (agent.id === "a") {
    const lines = [
      `${other.name}提到的「${hook}」有价值。我补可执行结构：指标、止损、小范围试验。下一轮请指出最脆弱的一条。`,
      `回应你刚才的点——我同意风险存在，但可被设计掉。对「${topic}」用反馈回路：观察→调整→再部署。`,
      `第${round}轮收敛：围绕「${hook}」建立可审计流程。请直接反驳“可审计”是否做得到。`,
    ];
    return lines[n % lines.length]!;
  }
  const lines = [
    `${other.name}，方案听起来干净，但现实会脏。围绕「${hook}」，激励会扭曲。先证明你能测到作弊。`,
    `你说“可执行”，我问执行者是谁、成本谁付、失败谁背锅。关于「${topic}」这三点说不清就是空转。`,
    `第${round}轮：默认乐观最可疑。${persona}要求压力测试——最坏是系统性失灵，不是稍差一点。`,
  ];
  return lines[n % lines.length]!;
}
