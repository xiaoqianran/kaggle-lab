# 015-dual-agent-chat · 双智对谈

两个智能体 **自动多轮对话**，走 **Kaggle Model Proxy**（Daily $10 / Monthly $100）。

与 `012-camel-roleplay` 的区别：

| | 012 CAMEL RolePlaying | **015 dual-agent-chat** |
|--|----------------------|-------------------------|
| 框架 | CAMEL `RolePlaying` | 裸 OpenAI 兼容 messages |
| 形态 | 任务驱动协作 | **自由对谈 / 辩论** |
| 人设 | 角色名 + task | A/B 完整 persona + topic |
| 产物 | 简报/计划 | 对谈 transcript（json/md） |

对应 Web 版「双智对谈」App 的同一套交互逻辑。

## 推荐模型（便宜好用）

```bash
python main.py 015 models
```

| 标签 | slug | 说明 |
|------|------|------|
| 推荐 | `google/gemini-3.5-flash-lite` | 默认，文笔自然 |
| 最便宜 | `google/gemini-3.1-flash-lite-preview` | 极限省额度 |
| 稳 | `openai/gpt-5.4-nano-2026-03-17` | 中文稳、讲理清楚 |

> 避免默认使用带大量 `reasoning_tokens` 的 `gemini-3-flash-preview`（易半截话 + 更贵）。

## 用法

```bash
source .venv/bin/activate
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"

# 凭证（约 1h）
python main.py 015 auth
# 或: python main.py 001 auth

# 冒烟 1 轮
python main.py 015 smoke

# 默认：激辩预设 × 4 轮
python main.py 015 run

# 预设 + 轮数 + 模型
python main.py 015 run --preset socratic --rounds 3
python main.py 015 run -m google/gemini-3.1-flash-lite-preview --rounds 2
python main.py 015 run --preset comedy -m openai/gpt-5.4-nano-2026-03-17

# 自定义话题/人设
python main.py 015 run --topic "开源模型会不会吃掉闭源？" \
  --a-name 开源信徒 --b-name 闭源猎手 --rounds 3
```

产物（不入库）：

- `015-dual-agent-chat/artifacts/last-dual-chat.json`
- `015-dual-agent-chat/artifacts/last-dual-chat.md`

## 预设

```bash
python main.py 015 presets
```

- `debate` 激辩 · 远程办公与创造力  
- `socratic` 苏格拉底 · 何为理解  
- `brainstorm` 脑暴 · 智能公寓  
- `comedy` 互怼 · 明天一定重构  

## 依赖

- 根目录 `.venv` + `kaggle` CLI + `openai`
- `KAGGLE_API_TOKEN` / `~/.kaggle/access_token`
- Model Proxy：`python main.py 001 auth` → `.env.model-proxy`

## Web UI（GitHub Pages + 本地网关）

浏览器**不能**直连 Kaggle Model Proxy（CORS）。网页版两种用法：

### A. GitHub Pages（静态）

部署后：`https://xiaoqianran.github.io/kaggle-lab/`

- **演示模式**：无需 key，本地模板对谈  
- **Live 模式**：粘贴 `MODEL_PROXY_API_KEY`，并把 **API Base** 指到可 CORS 的网关  
  （例如你本机/VPS 跑的 `gateway.py` 的 `https://xxx/api/openai`）

Actions：`.github/workflows/deploy-pages.yml`（push `015-dual-agent-chat/web/**` 或手动 workflow_dispatch）

### B. 本地一键（推荐 Live）

```bash
# 构建前端
cd 015-dual-agent-chat/web && npm i && npm run build && cd ../..

# 凭证
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
kaggle b auth -y --env-file .env.model-proxy

# 静态 + 反代
python 015-dual-agent-chat/gateway.py
# 打开 http://127.0.0.1:8765/
# 模式选 Live；API Base 默认/填 http://127.0.0.1:8765/api/openai
# Key 可留空（网关用 .env.model-proxy）
```

### 前端与 CLI 对齐

| 能力 | CLI `015 run` | Web |
|------|---------------|-----|
| 预设 / 模型三选一 | ✅ | ✅ |
| A↔B 自动轮 | ✅ | ✅ |
| Model Proxy | ✅ | ✅（经网关） |
| 产物 json/md | artifacts/ | 浏览器内 |

源码：`015-dual-agent-chat/web/`

### C. Cloudflare Worker（Pages 真 Live，推荐）

详见 **[cloudflare/README.md](./cloudflare/README.md)**。

```bash
cd 015-dual-agent-chat/cloudflare
npx wrangler login
npx wrangler secret put KAGGLE_API_TOKEN   # 粘贴 KGAT_…
npx wrangler deploy
```

Pages 里 Live → API Base = `https://<worker>.workers.dev/api/openai`，Key 留空。

