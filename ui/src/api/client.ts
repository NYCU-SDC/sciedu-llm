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

/** Rebuilds and resets are synchronous calls that re-index the whole corpus, so
 * they get a much longer leash than a listing. */
export const LONG_TIMEOUT_MS = 30 * 60 * 1000
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
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = text
    }
  }

  if (!response.ok) {
    const detail =
      payload && typeof payload === 'object' && 'detail' in payload
        ? (payload as { detail: unknown }).detail
        : payload
    throw new ApiError(response.status, detail, `${response.status} ${response.statusText}`)
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
