/* A small typed fetch wrapper.
 *
 * Every admin route answers errors as a FastAPI body: `{"detail": ...}`, where
 * `detail` is either a string (HTTPException) or a list of pydantic error
 * objects (a 422 from request-body validation). `ApiError` keeps both shapes so
 * a screen can render the flat message in an error panel *and*, for a 422,
 * point each problem at the field it came from. */

import type { ValidationProblem } from './errors'
import { detailToProblems, detailToMessage } from './errors'

/** Same-origin by default; the dev server proxies /admin — and /agents, which
 * the playground streams from — to :8080. Exported because the SSE client in
 * `agentsStream.ts` cannot go through `request()` (it reads the body itself)
 * but must resolve its URL exactly the same way. */
export const BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

const DEFAULT_TIMEOUT_MS = 30 * 1000

export class ApiError extends Error {
  readonly status: number
  readonly problems: ValidationProblem[]
  readonly detail: unknown

  constructor(status: number, detail: unknown, fallback: string) {
    super(detailToMessage(detail) || fallback)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.problems = detailToProblems(detail)
  }
}

/** A transport failure — the service did not answer at all. */
export class NetworkError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'NetworkError'
  }
}

interface RequestOptions {
  method?: string
  body?: unknown
  signal?: AbortSignal
  timeoutMs?: number
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, timeoutMs = DEFAULT_TIMEOUT_MS } = options

  // One controller for our own deadline, chained to any caller-supplied signal
  // so react-query can still cancel a query it no longer needs.
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const onAbort = () => controller.abort()
  signal?.addEventListener('abort', onAbort)

  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      signal: controller.signal,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch (error) {
    if (signal?.aborted) throw error
    if (controller.signal.aborted) {
      throw new NetworkError(
        `服務在 ${Math.round(timeoutMs / 1000)} 秒內未回應（${method} ${path}）。`,
      )
    }
    throw new NetworkError(
      `無法連線至 ${BASE || window.location.origin}${path} 的服務。` +
        `${error instanceof Error ? error.message : String(error)}`,
    )
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }

  if (response.status === 204) return undefined as T

  const text = await response.text()
  let payload: unknown = undefined
  // Whether the body actually was JSON. A failed parse is kept as raw text for
  // the error path below — an HTML 502 page from a proxy is worth showing — but
  // must never be handed back as if it were the typed body (see below).
  let isJson = true
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = text
      isJson = false
    }
  }

  if (!response.ok) {
    const detail =
      payload && typeof payload === 'object' && 'detail' in payload
        ? (payload as { detail: unknown }).detail
        : payload
    throw new ApiError(response.status, detail, `${response.status} ${response.statusText}`)
  }

  // A 2xx whose body is not JSON did not come from this API, however much it
  // looks like success: the usual cause is a proxy answering /admin itself —
  // nginx's SPA fallback serving index.html, or a gateway's interstitial page —
  // with 200. Returning that string as `T` is how a mis-set BACKEND_URL used to
  // surface as `TypeError: e.corpus_datasets is undefined` three layers into a
  // screen instead of as the routing problem it is.
  if (!isJson) {
    const excerpt = text.trim().replace(/\s+/g, ' ').slice(0, 120)
    throw new NetworkError(
      `${method} ${path} 回應了 ${response.status}，但內容不是 JSON——` +
        `請求可能沒有轉送到後端（檢查代理設定 / BACKEND_URL）。開頭是：${excerpt}`,
    )
  }

  return payload as T
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => request<T>(path, { signal }),
  post: <T>(path: string, body?: unknown, timeoutMs?: number) =>
    request<T>(path, { method: 'POST', body, timeoutMs }),
  put: <T>(path: string, body: unknown, timeoutMs?: number) =>
    request<T>(path, { method: 'PUT', body, timeoutMs }),
  patch: <T>(path: string, body: unknown, timeoutMs?: number) =>
    request<T>(path, { method: 'PATCH', body, timeoutMs }),
  delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
}

/** The Langfuse base URL, when the deployment told us about one at build time.
 * Every "open in Langfuse" affordance is hidden when this is unset rather than
 * guessed at. */
export const LANGFUSE_URL = (import.meta.env.VITE_LANGFUSE_URL ?? '').replace(/\/$/, '')

export function langfuseSessionUrl(sessionId: string): string | null {
  if (!LANGFUSE_URL) return null
  return `${LANGFUSE_URL}/sessions/${encodeURIComponent(sessionId)}`
}
