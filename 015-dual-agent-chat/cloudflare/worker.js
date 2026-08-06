/**
 * Cloudflare Worker: CORS gateway for Kaggle Model Proxy
 *
 * Browser (GitHub Pages) → this Worker → Kaggle Model Proxy
 *
 * Secrets (wrangler secret put / Dashboard):
 *   KAGGLE_API_TOKEN   required  (KGAT_…)
 *
 * Optional env (vars):
 *   ALLOW_ORIGIN       default *  (or https://xiaoqianran.github.io)
 *   GATEWAY_SECRET     if set, require header X-Gateway-Secret
 *
 * Routes used by 015 web:
 *   GET  /api/health
 *   POST /api/openai/chat/completions   (and other /api/openai/*)
 *   POST /api/auth/refresh              force refresh proxy token
 */

const KAGGLE_TOKEN_URL = "https://www.kaggle.com/api/v1/models/proxy/token";

/** @type {{ token: string, baseUri: string, expiryMs: number } | null} */
let cached = null;

export default {
  async fetch(request, env, ctx) {
    const origin = request.headers.get("Origin") || "*";
    const allowOrigin = env.ALLOW_ORIGIN || "*";
    const corsOrigin =
      allowOrigin === "*"
        ? "*"
        : origin && allowOrigin.split(",").map((s) => s.trim()).includes(origin)
          ? origin
          : allowOrigin.split(",")[0].trim();

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(corsOrigin) });
    }

    const url = new URL(request.url);

    try {
      if (env.GATEWAY_SECRET) {
        const got = request.headers.get("X-Gateway-Secret") || "";
        if (got !== env.GATEWAY_SECRET) {
          return json({ error: "unauthorized gateway secret" }, 401, corsOrigin);
        }
      }

      if (url.pathname === "/api/health" && request.method === "GET") {
        const hasTok = Boolean(env.KAGGLE_API_TOKEN);
        let proxyReady = false;
        let expiry = null;
        if (hasTok) {
          try {
            const p = await ensureProxy(env);
            proxyReady = Boolean(p.token);
            expiry = p.expiryMs ? new Date(p.expiryMs).toISOString() : null;
          } catch (e) {
            return json(
              { ok: false, service: "015-cf-worker", hasKaggleToken: hasTok, error: String(e) },
              503,
              corsOrigin,
            );
          }
        }
        return json(
          {
            ok: true,
            service: "015-cf-worker",
            hasKaggleToken: hasTok,
            proxyReady,
            expiry,
          },
          200,
          corsOrigin,
        );
      }

      if (url.pathname === "/api/auth/refresh" && request.method === "POST") {
        cached = null;
        const p = await fetchProxyToken(env);
        cached = p;
        return json(
          {
            ok: true,
            baseUri: p.baseUri,
            expiry: new Date(p.expiryMs).toISOString(),
          },
          200,
          corsOrigin,
        );
      }

      if (url.pathname.startsWith("/api/openai/")) {
        if (request.method !== "POST") {
          return json({ error: "POST only" }, 405, corsOrigin);
        }
        return await proxyOpenAI(request, env, corsOrigin, url.pathname);
      }

      return json(
        {
          error: "not found",
          hint: "Use POST /api/openai/chat/completions or GET /api/health",
        },
        404,
        corsOrigin,
      );
    } catch (e) {
      return json({ error: String(e?.message || e) }, 500, corsOrigin);
    }
  },
};

function corsHeaders(origin) {
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers":
      "Authorization, Content-Type, X-Gateway-Secret",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function json(obj, status, origin) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...corsHeaders(origin),
    },
  });
}

async function ensureProxy(env) {
  const now = Date.now();
  // refresh 3 min early
  if (cached && cached.expiryMs - now > 3 * 60 * 1000) {
    return cached;
  }
  cached = await fetchProxyToken(env);
  return cached;
}

async function fetchProxyToken(env) {
  const kgat = env.KAGGLE_API_TOKEN;
  if (!kgat) {
    throw new Error("Missing secret KAGGLE_API_TOKEN (KGAT_…)");
  }
  const res = await fetch(KAGGLE_TOKEN_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${kgat}`,
      "Content-Type": "application/json",
      "X-Kaggle-CLI-Source": "benchmarks-auth",
    },
    body: "{}",
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Kaggle token HTTP ${res.status}: ${t.slice(0, 200)}`);
  }
  const body = await res.json();
  const token = body.token || body.Token;
  const baseUri = (body.baseUri || body.base_uri || "").replace(/\/$/, "");
  const expiryRaw = body.expiryTime || body.expiry_time;
  const expiryMs = expiryRaw ? Date.parse(expiryRaw) : Date.now() + 50 * 60 * 1000;
  if (!token || !baseUri) {
    throw new Error("Kaggle token response missing token/baseUri");
  }
  return { token, baseUri, expiryMs };
}

async function proxyOpenAI(request, env, corsOrigin, pathname) {
  // Allow browser Authorization override; else use auto-refreshed proxy token
  let proxy = await ensureProxy(env);
  const authHeader = request.headers.get("Authorization") || "";
  let key = proxy.token;
  if (authHeader.toLowerCase().startsWith("bearer ") && authHeader.length > 20) {
    const maybe = authHeader.slice(7).trim();
    // if client sent empty-looking or "use-server", ignore
    if (maybe && maybe !== "use-server") {
      key = maybe;
    }
  }

  const sub = pathname.slice("/api/openai".length); // e.g. /chat/completions
  const target = `${proxy.baseUri}/openapi${sub}`;
  const body = await request.arrayBuffer();

  const doFetch = (apiKey) =>
    fetch(target, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body,
    });

  let res = await doFetch(key);
  if (res.status === 401 || res.status === 403) {
    cached = null;
    proxy = await ensureProxy(env);
    res = await doFetch(proxy.token);
  }

  const data = await res.arrayBuffer();
  const headers = new Headers(corsHeaders(corsOrigin));
  headers.set(
    "Content-Type",
    res.headers.get("Content-Type") || "application/json",
  );
  return new Response(data, { status: res.status, headers });
}
