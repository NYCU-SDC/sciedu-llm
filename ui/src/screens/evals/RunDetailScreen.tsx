import { Link, useNavigate, useParams } from 'react-router-dom'
import { ExternalLink } from 'lucide-react'

import { langfuseSessionUrl } from '../../api/client'
import { errorMessage } from '../../api/errors'
import { useEvalHistory, useEvalMutations, useEvalRun } from '../../api/hooks'
import type { EvalRun } from '../../api/types'
import { isTerminal } from '../../api/types'
import { CopyButton } from '../../components/CopyButton'
import { ErrorPanel, QueryError } from '../../components/ErrorPanel'
import { Panel } from '../../components/Panel'
import { Loading, PageHeader } from '../../components/States'
import { RunStatusTag } from '../../components/StatusTag'
import { formatDateTime, formatDuration, joinNames } from '../../lib/format'
import type { EvalPrefill } from './EvalsScreen'

export function RunDetailScreen() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const run = useEvalRun(runId)
  const { cancel } = useEvalMutations()

  if (run.isError) {
    return (
      <>
        <PageHeader title={runId ?? '執行紀錄'} back={<BackLink />} mono />
        <div style={{ marginTop: 20 }}>
          <QueryError
            what={`無法開啟執行紀錄「${runId}」`}
            error={run.error}
            actions={
              <Link className="btn btn-primary" to="/evals">
                返回執行清單
              </Link>
            }
          />
          <p className="note" style={{ marginTop: 10 }}>
            執行清單儲存在服務記憶體中。重新啟動會清空清單，因此重啟前的執行 ID 將無法找到；其追蹤紀錄仍在 Langfuse。
          </p>
        </div>
      </>
    )
  }

  // Checked after `isError`, so a failure never reads as a slow load.
  if (!run.data) {
    return (
      <>
        <PageHeader title={runId ?? '執行紀錄'} back={<BackLink />} mono />
        <Loading what="執行紀錄" />
      </>
    )
  }

  const data = run.data
  const live = !isTerminal(data.status)
  const sessionUrl = data.session_id ? langfuseSessionUrl(data.session_id) : null

  const runAgain = () => {
    const prefill: EvalPrefill = {
      eval_model: data.eval_model,
      judge_model: data.judge_model,
      corpus_datasets: data.corpus_datasets,
      question_datasets: data.question_datasets,
      judge_prompts: data.judge_prompts,
      k: data.k,
      embedding_model: data.embedding_model,
      rerank_model: data.rerank_model,
      chunk_size: data.chunk_size,
      chunk_overlap: data.chunk_overlap,
    }
    void navigate('/evals', { state: { prefill } })
  }

  return (
    <>
      <PageHeader
        back={<BackLink />}
        title={data.run_id}
        mono
        lede={<Timing run={data} />}
        actions={
          <>
            <button type="button" className="btn btn-secondary" onClick={runAgain}>
              使用這些設定再次執行
            </button>
            {live && (
              <button
                type="button"
                className="btn btn-secondary btn-danger"
                onClick={() => cancel.mutate(data.run_id)}
                disabled={cancel.isPending}
              >
                {cancel.isPending ? '取消中…' : '取消此次執行'}
              </button>
            )}
            {sessionUrl && (
              <a className="btn btn-primary" href={sessionUrl} target="_blank" rel="noreferrer">
                在 Langfuse 開啟追蹤紀錄
                <ExternalLink size={14} strokeWidth={2.75} aria-hidden />
              </a>
            )}
          </>
        }
      />

      {live && (
        <div className="banner banner-busy" style={{ marginTop: 20 }} aria-live="polite">
          <span className="banner-led" />
          <div className="banner-body">
            <div className="banner-title">
              {data.status === 'building'
                ? '正在為此次執行建立專用索引'
                : data.status === 'judging'
                  ? '正在回答題目並評分'
                  : '正在等待開始'}
            </div>
            <div className="banner-line">
              此頁面每隔數秒重新讀取執行狀態。服務只回報階段，不回報百分比。
            </div>
          </div>
          <RunStatusTag status={data.status} />
        </div>
      )}

      {data.status === 'failed' && data.error && (
        <div style={{ marginTop: 20 }}>
          <ErrorPanel
            title="此次執行失敗"
            detail={data.error}
            copyText={data.error}
            actions={
              <button type="button" className="btn btn-primary" onClick={runAgain}>
                使用這些設定再試一次
              </button>
            }
          />
        </div>
      )}

      {cancel.error && (
        <div style={{ marginTop: 14 }}>
          <QueryError what="無法取消此次執行" error={cancel.error} />
        </div>
      )}

      <div className="split split-wide" style={{ marginTop: 20 }}>
        <div className="col">
          <Panel title="此次執行使用的設定">
            <table className="table">
              <tbody>
                <ParamRow label="狀態" value={<RunStatusTag status={data.status} />} />
                <ParamRow label="受測模型" value={data.eval_model} mono />
                <ParamRow label="評分模型" value={data.judge_model} mono />
                <ParamRow
                  label="課程教材"
                  value={joinNames(data.corpus_datasets)}
                  mono
                />
                <ParamRow label="題目集" value={joinNames(data.question_datasets)} mono />
                <ParamRow label="評分提示詞" value={joinNames(data.judge_prompts)} mono />
                <ParamRow label="每題段落數（k）" value={String(data.k)} mono />
                <ParamRow
                  label="嵌入／重排序"
                  value={`${data.embedding_model} / ${data.rerank_model}`}
                  mono
                />
                <ParamRow
                  label="片段切分"
                  value={`${data.chunk_size} / ${data.chunk_overlap}`}
                  mono
                />
                <ParamRow label="最大並行數" value={String(data.max_concurrency)} mono />
                <ParamRow label="開始時間" value={formatDateTime(data.started_at)} mono />
                <ParamRow
                  label="完成時間"
                  value={data.finished_at ? formatDateTime(data.finished_at) : '仍在執行中'}
                  mono
                />
                <ParamRow
                  label="耗時"
                  value={formatDuration(data.duration_seconds)}
                  mono
                />
              </tbody>
            </table>
          </Panel>

          {data.question_datasets.map((dataset) => (
            <HistoryPanel key={dataset} dataset={dataset} />
          ))}
        </div>

        <aside className="sticky-side">
          <Panel title="Langfuse 工作階段">
            {data.session_id ? (
              <>
                <div className="mono" style={{ fontSize: 12.5, wordBreak: 'break-all' }}>
                  {data.session_id}
                </div>
                <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                  <CopyButton text={data.session_id} label="複製工作階段 ID" />
                  {sessionUrl && (
                    <a className="btn btn-primary" href={sessionUrl} target="_blank" rel="noreferrer">
                      在 Langfuse 開啟
                      <ExternalLink size={14} strokeWidth={2.75} aria-hidden />
                    </a>
                  )}
                </div>
                <p className="note" style={{ marginTop: 10 }}>
                  每個問題、檢索到的段落、答案與各項分數都會記錄在這個工作階段 ID 下。
                  {!sessionUrl &&
                    ' 此版本未設定 Langfuse 位址，因此沒有連結；請將 ID 貼到您的 Langfuse。'}
                </p>
              </>
            ) : (
              <p className="note">
                尚無工作階段 ID。執行送達 Langfuse 後，服務便會記錄一個 ID。
              </p>
            )}
          </Panel>

          <div
            style={{
              background: 'var(--color-accent-2-100)',
              border: '1px solid var(--color-accent-2-300)',
              borderRadius: 12,
              padding: '14px 16px',
              fontSize: 12.5,
              lineHeight: 1.55,
              color: 'var(--color-accent-2-800)',
            }}
          >
            <strong>分數保存在 Langfuse。</strong>此服務只在記憶體中保存執行狀態與設定，不會彙整分數；請在上方開啟工作階段查看。
          </div>
        </aside>
      </div>
    </>
  )
}

function BackLink() {
  return (
    <Link to="/evals" style={{ fontSize: 12.5 }}>
      ← 所有執行紀錄
    </Link>
  )
}

function Timing({ run }: { run: EvalRun }) {
  return (
    <span style={{ fontSize: 13 }}>
      開始於 {formatDateTime(run.started_at)}
      {run.finished_at
        ? ` · 完成於 ${formatDateTime(run.finished_at)}`
        : ' · 仍在執行中'}{' '}
      · 耗時 {formatDuration(run.duration_seconds)}
    </span>
  )
}

function ParamRow({
  label,
  value,
  mono,
}: {
  label: string
  value: React.ReactNode
  mono?: boolean
}) {
  return (
    <tr>
      <td style={{ width: 220, fontSize: 12.5, color: 'var(--color-neutral-700)' }}>
        {label}
      </td>
      <td className={mono ? 'mono' : undefined} style={{ fontSize: 12.5 }}>
        {value}
      </td>
    </tr>
  )
}

/** Langfuse's own record of past judge runs against this question dataset —
 * durable, unlike the in-memory run list. It is a separate upstream call, so it
 * fails on its own without taking the page with it. */
function HistoryPanel({ dataset }: { dataset: string }) {
  const history = useEvalHistory(dataset)
  return (
    <Panel title={<>{dataset} 的較早執行紀錄</>}>
      <p className="note" style={{ marginBottom: 10 }}>
        從 Langfuse 讀取，因此此清單在重新啟動後仍會保留。
      </p>
      {history.isError ? (
        <p className="note" style={{ color: 'var(--color-alarm-ink)' }}>
          無法讀取 Langfuse — {errorMessage(history.error)}{' '}
          <button type="button" className="link-btn" onClick={() => void history.refetch()}>
            再試一次
          </button>
        </p>
      ) : !history.data ? (
        <Loading what="歷史紀錄" />
      ) : history.data.length === 0 ? (
        <p className="note">Langfuse 尚未記錄此資料集的執行紀錄。</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>實驗</th>
              <th>時間</th>
              <th>備註</th>
            </tr>
          </thead>
          <tbody>
            {history.data?.map((entry) => (
              <tr key={`${entry.dataset_name}:${entry.run_name}`}>
                <td className="mono" style={{ fontSize: 12.5 }}>
                  {entry.run_name}
                </td>
                <td className="mono" style={{ fontSize: 12.5 }}>
                  {formatDateTime(entry.created_at)}
                </td>
                <td style={{ fontSize: 12.5, color: 'var(--color-neutral-700)' }}>
                  {entry.description ?? '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  )
}
