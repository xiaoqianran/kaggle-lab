# Cloudflare Worker 网关（给 GitHub Pages Live 用）

浏览器不能直连 Kaggle Model Proxy（CORS）。  
用 **Cloudflare Worker** 做中转：自动 `KGAT` → 临时 Proxy Key → 转发 chat。

```text
GitHub Pages UI
   → https://<你的worker>.workers.dev/api/openai/chat/completions
   → Worker（自动刷 token + CORS）
   → https://mp-….kaggle.net/models/openapi/...
```

## 1. 准备

- 免费 [Cloudflare](https://dash.cloudflare.com/) 账号  
- 本机 Node 18+（用 Dashboard 粘贴也可，不必装 wrangler）  
- Kaggle API Token：`KGAT_…`（和 `kaggle b auth` 同一个）

## 2A. 命令行部署（推荐）

```bash
cd 015-dual-agent-chat/cloudflare
npm i -g wrangler   # 或 npx wrangler

npx wrangler login
npx wrangler secret put KAGGLE_API_TOKEN
# 粘贴 KGAT_… 后回车

# 可选：限制调用
# npx wrangler secret put GATEWAY_SECRET

npx wrangler deploy
```

部署成功会打印类似：

```text
https://kaggle-lab-015-gateway.<你的子域>.workers.dev
```

自检：

```bash
curl -s https://kaggle-lab-015-gateway.<子域>.workers.dev/api/health
```

应看到 `"ok":true,"proxyReady":true`。

## 2B. Dashboard 点点点（不装 CLI）

1. Cloudflare Dashboard → **Workers & Pages** → **Create** → **Worker**  
2. 编辑代码：把本目录 `worker.js` **全文粘贴**进去（若编辑器是 `export default` 模块格式，CF 新编辑器支持；若只支持 `addEventListener`，用下面「旧式」或继续用 wrangler）  
3. **Settings → Variables** → 添加 Secret：  
   - `KAGGLE_API_TOKEN` = `KGAT_…`  
4. **Deploy**  
5. 记下 `*.workers.dev` 地址  

> 若 Dashboard 模板是 `addEventListener('fetch')` 老格式，请用 **wrangler deploy**（本仓库 worker 为 modules 格式）。

## 3. 接到 015 网页

打开：https://xiaoqianran.github.io/kaggle-lab/

1. 侧边栏打开 **凭证 / Live**  
2. 模式选 **Live**  
3. **API Base** 填：

```text
https://kaggle-lab-015-gateway.<你的子域>.workers.dev/api/openai
```

4. **API Key** 可留空（Worker 用 Secret 自动刷）  
5. 若设了 `GATEWAY_SECRET`：需要前端带 `X-Gateway-Secret`（当前 UI 未做该框时，可先不设 Secret，或只给自己用）  
6. 开始对谈  

## 4. 安全（必读）

| 风险 | 建议 |
|------|------|
| `*.workers.dev` 公网谁都能打 → **烧你的 Kaggle 额度** | 设 `GATEWAY_SECRET`，或 `ALLOW_ORIGIN=https://xiaoqianran.github.io` |
| Secret 写进前端 | **永远不要**；只放 Worker Secrets |
| KGAT 泄露 | 立刻在 Kaggle 作废重发 |

限制来源示例（`wrangler.toml`）：

```toml
[vars]
ALLOW_ORIGIN = "https://xiaoqianran.github.io"
```

## 5. 和本机 gateway.py 对比

| | `gateway.py` | **CF Worker** |
|--|--------------|---------------|
| Pages Live | 不行（127.0.0.1） | ✅ |
| 自动刷 1h token | 需本机 `kaggle b auth` | ✅ Worker 内自动 |
| 费用 | 免费本机 | CF 免费额度通常够玩 |

## 6. 故障

| 现象 | 处理 |
|------|------|
| health 里 missing KAGGLE_API_TOKEN | `wrangler secret put KAGGLE_API_TOKEN` |
| token HTTP 401 | KGAT 错/过期，Kaggle 账户重发 |
| 聊天 CORS | API Base 必须是 Worker 的 `/api/openai`，不要填 mp-staging 直链 |
| 401 from proxy | 等 Worker 自动 refresh，或 POST `/api/auth/refresh` |
