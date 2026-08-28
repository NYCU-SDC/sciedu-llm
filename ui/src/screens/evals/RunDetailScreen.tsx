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
        <PageHeader title={runId ?? 'Run'} back={<BackLink />} mono />
        <div style={{ marginTop: 20 }}>
          <QueryError
            what={`Could not open run '${runId}'`}
            error={run.error}
            actions={
              <Link className="btn btn-primary" to="/evals">
                Back to the run list
              </Link>
            }
          />
          <p className="note" style={{ marginTop: 10 }}>
            The run list lives in the service's memory. A restart clears it, and a run id
            from before the restart will not be found — its traces are still in Langfuse.
          </p>
        </div>
      </>
    )
  }

  // Checked after `isError`, so a failure never reads as a slow load.
  if (!run.data) {
    return (
      <>
        <PageHeader title={runId ?? 'Run'} back={<BackLink />} mono />
        <Loading what="the run" />
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
              Run again with these settings
            </button>
            {live && (
              <button
                type="button"
                className="btn btn-secondary btn-danger"
                onClick={() => cancel.mutate(data.run_id)}
                disabled={cancel.isPending}
              >
                {cancel.isPending ? 'Cancelling…' : 'Cancel this run'}
              </button>
            )}
            {sessionUrl && (
              <a className="btn btn-primary" href={sessionUrl} target="_blank" rel="noreferrer">
                Open traces in Langfuse
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
                ? 'Building a private index for this run'
                : data.status === 'judging'
                  ? 'Answering and scoring the questions'
                  : 'Waiting to start'}
            </div>
            <div className="banner-line">
              This page re-reads the run every few seconds. The service reports the stage,
              not a percentage.
            </div>
          </div>
          <RunStatusTag status={data.status} />
        </div>
      )}

      {data.status === 'failed' && data.error && (
        <div style={{ marginTop: 20 }}>
          <ErrorPanel
            title="This run failed"
            detail={data.error}
            copyText={data.error}
            actions={
              <button type="button" className="btn btn-primary" onClick={runAgain}>
                Try again with these settings
              </button>
            }
          />
        </div>
      )}

      {cancel.error && (
        <div style={{ marginTop: 14 }}>
          <QueryError what="Could not cancel this run" error={cancel.error} />
        </div>
      )}

      <div className="split split-wide" style={{ marginTop: 20 }}>
        <div className="col">
          <Panel title="Settings this run used">
            <table className="table">
              <tbody>
                <ParamRow label="Status" value={<RunStatusTag status={data.status} />} />
                <ParamRow label="Model tested" value={data.eval_model} mono />
                <ParamRow label="Model scoring" value={data.judge_model} mono />
                <ParamRow
                  label="Course material"
                  value={joinNames(data.corpus_datasets)}
                  mono
                />
                <ParamRow label="Question sets" value={joinNames(data.question_datasets)} mono />
                <ParamRow label="Scoring prompts" value={joinNames(data.judge_prompts)} mono />
                <ParamRow label="Passages per question (k)" value={String(data.k)} mono />
                <ParamRow
                  label="Embedding / rerank"
                  value={`${data.embedding_model} / ${data.rerank_model}`}
                  mono
                />
                <ParamRow
                  label="Chunking"
                  value={`${data.chunk_size} / ${data.chunk_overlap}`}
                  mono
                />
                <ParamRow label="Max concurrency" value={String(data.max_concurrency)} mono />
                <ParamRow label="Started" value={formatDateTime(data.started_at)} mono />
                <ParamRow
                  label="Finished"
                  value={data.finished_at ? formatDateTime(data.finished_at) : 'still running'}
                  mono
                />
                <ParamRow
                  label="Duration"
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
          <Panel title="Langfuse session">
            {data.session_id ? (
              <>
                <div className="mono" style={{ fontSize: 12.5, wordBreak: 'break-all' }}>
                  {data.session_id}
                </div>
                <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                  <CopyButton text={data.session_id} label="Copy session id" />
                  {sessionUrl && (
                    <a className="btn btn-primary" href={sessionUrl} target="_blank" rel="noreferrer">
                      Open in Langfuse
                      <ExternalLink size={14} strokeWidth={2.75} aria-hidden />
                    </a>
                  )}
                </div>
                <p className="note" style={{ marginTop: 10 }}>
                  Every question, the passages it retrieved, the answer and each score are
                  there under this session id.
                  {!sessionUrl &&
                    ' This build was not told where Langfuse lives, so there is no link — paste the id into your own Langfuse.'}
                </p>
              </>
            ) : (
              <p className="note">
                No session id yet. The service records one once the run reaches Langfuse.
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
            <strong>Scores live in Langfuse.</strong> This service keeps only the run's
            status and settings in memory — it does not aggregate the scores, so there is
            nothing honest to show here. Open the session above to read them.
          </div>
        </aside>
      </div>
    </>
  )
}

function BackLink() {
  return (
    <Link to="/evals" style={{ fontSize: 12.5 }}>
      ← All runs
    </Link>
  )
}

function Timing({ run }: { run: EvalRun }) {
  return (
    <span style={{ fontSize: 13 }}>
      Started {formatDateTime(run.started_at)}
      {run.finished_at
        ? ` · finished ${formatDateTime(run.finished_at)}`
        : ' · still running'}{' '}
      · took {formatDuration(run.duration_seconds)}
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
    <Panel title={<>Earlier runs on {dataset}</>}>
      <p className="note" style={{ marginBottom: 10 }}>
        Read back from Langfuse, so this list survives restarts.
      </p>
      {history.isError ? (
        <p className="note" style={{ color: 'var(--color-alarm-ink)' }}>
          Langfuse could not be read — {errorMessage(history.error)}{' '}
          <button type="button" className="link-btn" onClick={() => void history.refetch()}>
            try again
          </button>
        </p>
      ) : !history.data ? (
        <Loading what="the history" />
      ) : history.data.length === 0 ? (
        <p className="note">Langfuse has no recorded runs against this dataset yet.</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Experiment</th>
              <th>When</th>
              <th>Notes</th>
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
