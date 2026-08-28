import { useMemo, useState } from 'react'

import { ApiError } from '../../api/client'
import { errorMessage } from '../../api/errors'
import { useDatasets, useModels, useRagConfig, useRagMutations } from '../../api/hooks'
import { CheckList, type Choice } from '../../components/Choices'
import { ConfirmDialog } from '../../components/ConfirmDialog'
import { ErrorPanel, QueryError } from '../../components/ErrorPanel'
import { Field, Panel } from '../../components/Panel'
import { Loading, PageHeader } from '../../components/States'
import { RebuildTag } from '../../components/StatusTag'
import { pluralise } from '../../lib/format'
import { buildUpdate, diffDraft, toDraft, type NumericKey, type RagDraft } from './draft'

/** The plain-language notes the mockup puts beside each retrieval knob. */
const KNOBS: { key: NumericKey; label: string; note: string }[] = [
  {
    key: 'bm25_top_n',
    label: 'Keyword top-n',
    note: 'Passages kept from the exact-wording search (BM25).',
  },
  {
    key: 'dense_top_n',
    label: 'Meaning top-n',
    note: 'Passages kept from the similar-meaning search.',
  },
  {
    key: 'rrf_k',
    label: 'Merge constant (RRF k)',
    note: 'How gently the two lists are blended. Higher is gentler.',
  },
  {
    key: 'rerank_pool_size',
    label: 'Rerank pool',
    note: 'How many of the merged passages get re-read and re-ordered.',
  },
  {
    key: 'final_k',
    label: 'Final k',
    note: 'How many passages the assistant actually reads before answering.',
  },
]

export function RagScreen() {
  const config = useRagConfig()
  const datasets = useDatasets()
  const models = useModels()
  const { apply, rebuild, reset } = useRagMutations()

  const [draft, setDraft] = useState<RagDraft | null>(null)
  const [confirmReset, setConfirmReset] = useState(false)

  const server = useMemo(() => (config.data ? toDraft(config.data) : null), [config.data])

  // The draft is seeded from the server snapshot and only exists once the user
  // has touched something; that way a background refetch cannot clobber an edit
  // in progress, and "no draft" genuinely means "no unsaved changes".
  const current = draft ?? server
  const changes = useMemo(
    () => (server && draft ? diffDraft(server, draft) : []),
    [server, draft],
  )
  const built = useMemo(
    () => (draft ? buildUpdate(changes, draft) : null),
    [changes, draft],
  )

  const busy = apply.isPending || rebuild.isPending || reset.isPending
  const rebuildCount = changes.filter((change) => change.rebuilds).length
  const rebuildsOnApply = rebuildCount > 0

  const set = <K extends keyof RagDraft>(key: K, value: RagDraft[K]) => {
    setDraft((previous) => {
      const base = previous ?? server
      if (!base) return previous
      return { ...base, [key]: value }
    })
  }

  const problemFor = (key: keyof RagDraft) =>
    built?.problems.find((problem) => problem.key === key)?.message

  const onApply = () => {
    if (!built || built.problems.length > 0) return
    apply.mutate(
      { ...built.update, rebuild: rebuildsOnApply },
      { onSuccess: () => setDraft(null) },
    )
  }

  if (config.isError) {
    const notEnabled = config.error instanceof ApiError && config.error.status === 503
    return (
      <>
        <Header disabled onReset={() => undefined} onRebuild={() => undefined} />
        <div style={{ marginTop: 20 }}>
          {notEnabled ? (
            <ErrorPanel
              title="Retrieval is switched off on this service"
              detail={errorMessage(config.error)}
            >
              <p className="alarm-body" style={{ marginTop: 10 }}>
                Set <span className="mono">RAG_CORPUS_DATASETS</span> in the service's
                environment and restart it; there is nothing to configure until then.
              </p>
            </ErrorPanel>
          ) : (
            <QueryError
              what="Could not read the retrieval configuration"
              error={config.error}
              actions={
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => void config.refetch()}
                >
                  Try again
                </button>
              }
            />
          )}
        </div>
      </>
    )
  }

  // No error and no snapshot yet: still on the way. Deliberately checked after
  // `isError`, so a failure is never mistaken for a slow load.
  if (!config.data || !current) {
    return (
      <>
        <Header disabled onReset={() => undefined} onRebuild={() => undefined} />
        <Loading what="the live retrieval configuration" />
      </>
    )
  }

  const live = config.data
  const corpusChoices = buildCorpusChoices(
    datasets.data?.corpus.map((entry) => entry.name) ?? [],
    live.corpus_datasets,
  )
  const modelOptions = models.data?.models ?? []

  return (
    <>
      <Header
        disabled={busy}
        onReset={() => setConfirmReset(true)}
        onRebuild={() => rebuild.mutate()}
      />

      <StatusBanner
        isBuilt={live.is_built}
        datasetCount={live.corpus_datasets.length}
        pending={
          apply.isPending
            ? rebuildsOnApply
              ? 'Applying your changes, then rebuilding the index'
              : 'Applying your changes'
            : rebuild.isPending
              ? 'Rebuilding the index'
              : reset.isPending
                ? 'Restoring the server defaults, then rebuilding the index'
                : null
        }
        failure={
          apply.error ?? rebuild.error ?? reset.error
            ? errorMessage(apply.error ?? rebuild.error ?? reset.error)
            : null
        }
        onRetry={() => rebuild.mutate()}
      />

      <div className="split split-wide" style={{ marginTop: 22 }}>
        <div className="col">
          <Panel title="Course material">
            <p className="note" style={{ marginBottom: 14 }}>
              Which Langfuse datasets under <span className="mono">corpus/</span> the
              assistant may quote from.
            </p>
            {datasets.isError ? (
              <QueryError
                what="Could not list the Langfuse datasets"
                error={datasets.error}
                actions={
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => void datasets.refetch()}
                  >
                    Try again
                  </button>
                }
              />
            ) : !datasets.data ? (
              <Loading what="the dataset list" />
            ) : (
              <CheckList
                choices={corpusChoices}
                selected={current.corpus_datasets}
                disabled={busy}
                empty="Langfuse has no datasets under the corpus folder yet."
                onToggle={(value, next) => {
                  const chosen = new Set(current.corpus_datasets)
                  if (next) chosen.add(value)
                  else chosen.delete(value)
                  set('corpus_datasets', [...chosen].sort())
                }}
              />
            )}
            {problemFor('corpus_datasets') && (
              <p className="note" style={{ marginTop: 10, color: 'var(--color-alarm-ink)' }}>
                {problemFor('corpus_datasets')}
              </p>
            )}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                marginTop: 12,
                flexWrap: 'wrap',
              }}
            >
              <RebuildTag />
              <span style={{ fontSize: 12, color: 'var(--color-neutral-600)' }}>
                Read from Langfuse ·{' '}
                <button
                  type="button"
                  className="link-btn"
                  onClick={() => void datasets.refetch()}
                >
                  refresh the list
                </button>
              </span>
            </div>
          </Panel>

          <Panel title="Models">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div className="grid-2" style={{ alignItems: 'start' }}>
                <div>
                  <Field label="Embedding model">
                    {(id) => (
                      <ModelPicker
                        id={id}
                        value={current.embedding_model}
                        options={modelOptions}
                        disabled={busy}
                        onChange={(value) => set('embedding_model', value)}
                      />
                    )}
                  </Field>
                  <div style={{ marginTop: 7 }}>
                    <RebuildTag>rebuilds the index</RebuildTag>
                  </div>
                </div>
                <p className="note" style={{ paddingTop: 19 }}>
                  Turns your documents into numbers so passages with similar meaning can be
                  found. Changing it means every document has to be re-read.
                </p>
              </div>
              <div className="grid-2" style={{ alignItems: 'start' }}>
                <Field label="Rerank model">
                  {(id) => (
                    <ModelPicker
                      id={id}
                      value={current.rerank_model}
                      options={modelOptions}
                      disabled={busy}
                      onChange={(value) => set('rerank_model', value)}
                    />
                  )}
                </Field>
                <p className="note" style={{ paddingTop: 19 }}>
                  A second pass that re-orders the candidate passages by how well they
                  answer the question. Safe to change at any time.
                </p>
              </div>
            </div>
            {models.isError && (
              <p className="note" style={{ marginTop: 12 }}>
                The model list is unavailable ({errorMessage(models.error)}), so these are
                free-text fields for now.
              </p>
            )}
          </Panel>

          <Panel title="How documents are cut up">
            <p className="note" style={{ marginBottom: 14 }}>
              Long documents are split into overlapping pieces. Smaller pieces are more
              precise; larger pieces keep more context around each fact.
            </p>
            <div className="grid-2">
              <NumberField
                label="Chunk size (characters)"
                value={current.chunk_size}
                problem={problemFor('chunk_size')}
                disabled={busy}
                onChange={(value) => set('chunk_size', value)}
              />
              <NumberField
                label="Chunk overlap (characters)"
                value={current.chunk_overlap}
                problem={problemFor('chunk_overlap')}
                disabled={busy}
                onChange={(value) => set('chunk_overlap', value)}
              />
            </div>
            <div style={{ marginTop: 12 }}>
              <RebuildTag>changing these rebuilds the index</RebuildTag>
            </div>
          </Panel>

          <Panel title="How passages are found">
            <p className="note" style={{ marginBottom: 14 }}>
              Two searches run side by side — keyword and meaning — and their results are
              merged, trimmed and re-ordered before the assistant sees them.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {KNOBS.map((knob) => (
                <div className="knob-row" key={knob.key}>
                  <label htmlFor={`knob-${knob.key}`}>{knob.label}</label>
                  <input
                    id={`knob-${knob.key}`}
                    className={`input mono input-sunken${problemFor(knob.key) ? ' input-invalid' : ''}`}
                    inputMode="numeric"
                    value={current[knob.key]}
                    disabled={busy}
                    onChange={(event) => set(knob.key, event.target.value)}
                  />
                  <span className="note">
                    {knob.note}
                    {problemFor(knob.key) && (
                      <span style={{ display: 'block', color: 'var(--color-alarm-ink)' }}>
                        {problemFor(knob.key)}
                      </span>
                    )}
                  </span>
                </div>
              ))}
            </div>
          </Panel>

          <div className="grid-2" style={{ gap: 14 }}>
            <Panel title="Answer prompts">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <TextField
                  label="System prompt (Langfuse)"
                  value={current.generator_system_prompt_name}
                  problem={problemFor('generator_system_prompt_name')}
                  disabled={busy}
                  onChange={(value) => set('generator_system_prompt_name', value)}
                />
                <TextField
                  label="User prompt (Langfuse)"
                  value={current.generator_user_prompt_name}
                  problem={problemFor('generator_user_prompt_name')}
                  disabled={busy}
                  onChange={(value) => set('generator_user_prompt_name', value)}
                />
              </div>
              <p className="note" style={{ marginTop: 12 }}>
                The wording lives in Langfuse; this only says which prompt to fetch.
              </p>
            </Panel>
            <Panel title="Throughput">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <NumberField
                  label="Embedding batch size"
                  value={current.embedding_batch_size}
                  problem={problemFor('embedding_batch_size')}
                  disabled={busy}
                  onChange={(value) => set('embedding_batch_size', value)}
                />
                <NumberField
                  label="Max concurrency"
                  value={current.max_concurrency}
                  problem={problemFor('max_concurrency')}
                  disabled={busy}
                  onChange={(value) => set('max_concurrency', value)}
                />
              </div>
              <p className="note" style={{ marginTop: 12 }}>
                How hard a rebuild pushes the model server. Lower these if rebuilds fail
                with rate-limit errors.
              </p>
            </Panel>
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 18,
              flexWrap: 'wrap',
              padding: '4px 2px',
            }}
          >
            <button
              type="button"
              className="btn btn-primary"
              style={{ fontSize: 14.5, padding: '11px 22px' }}
              disabled={busy || changes.length === 0 || (built?.problems.length ?? 0) > 0}
              onClick={onApply}
            >
              {apply.isPending ? 'Applying…' : 'Apply to the running service'}
            </button>
            <p className="note" style={{ maxWidth: '46ch' }}>
              Applies immediately, and only in memory —{' '}
              <strong>
                if the service restarts, these values go back to the server's defaults.
              </strong>{' '}
              {changes.length === 0
                ? 'Nothing is waiting to be applied.'
                : rebuildsOnApply
                  ? `${rebuildCount === 1 ? 'One of your changes needs' : `${rebuildCount} of your changes need`} the index rebuilt, which takes a while.`
                  : 'None of your changes need the index rebuilt.'}
            </p>
          </div>
        </div>

        <aside className="panel" style={{ position: 'sticky', top: 26 }}>
          <h5 className="sect" style={{ margin: 0 }}>
            Unsaved changes
          </h5>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 12 }}>
            {changes.length === 0 ? (
              <p className="note">
                Nothing edited yet. What you see is exactly what the service is using.
              </p>
            ) : (
              <>
                {changes.map((change) => (
                  <div className="tile" key={change.key}>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>{change.label}</div>
                    <div className="mono" style={{ fontSize: 12, marginTop: 1 }}>
                      <span
                        style={{
                          textDecoration: 'line-through',
                          color: 'var(--color-neutral-500)',
                        }}
                      >
                        {change.from}
                      </span>{' '}
                      →{' '}
                      <span style={{ color: 'var(--color-accent-700)', fontWeight: 600 }}>
                        {change.to}
                      </span>
                    </div>
                    {change.rebuilds && (
                      <div style={{ marginTop: 6 }}>
                        <RebuildTag>rebuilds the index</RebuildTag>
                      </div>
                    )}
                  </div>
                ))}
                <p className="note">Nothing is sent to the service until you press Apply.</p>
                <button
                  type="button"
                  className="btn btn-ghost"
                  style={{ alignSelf: 'flex-start' }}
                  disabled={busy}
                  onClick={() => setDraft(null)}
                >
                  Discard changes
                </button>
              </>
            )}
          </div>
        </aside>
      </div>

      {confirmReset && (
        <ConfirmDialog
          title="Reset to the server's defaults?"
          body={
            <>
              Every override this process is holding is dropped and the values from the
              service's <span className="mono">RAG_*</span> environment variables come back.
              The index is rebuilt straight afterwards, which can take several minutes.
            </>
          }
          confirmLabel="Reset and rebuild"
          busy={reset.isPending}
          onCancel={() => setConfirmReset(false)}
          onConfirm={() => {
            setConfirmReset(false)
            reset.mutate(undefined, { onSuccess: () => setDraft(null) })
          }}
        />
      )}
    </>
  )
}

function Header({
  disabled,
  onReset,
  onRebuild,
}: {
  disabled: boolean
  onReset: () => void
  onRebuild: () => void
}) {
  return (
    <PageHeader
      kicker="Live service"
      title="Retrieval settings"
      lede="How the assistant looks things up in your course material before it answers. Every setting below is in use right now."
      actions={
        <>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={disabled}
            onClick={onReset}
          >
            Reset to server defaults
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={disabled}
            onClick={onRebuild}
          >
            Rebuild index
          </button>
        </>
      }
    />
  )
}

/** Honest states only. The service reports `is_built` and, on a failure, the
 * reason — it does not report chunk counts, build timings or embedding
 * progress, so none of those appear here. A rebuild in flight gets a pulsing
 * dot rather than a progress bar that would have to be invented. */
function StatusBanner({
  isBuilt,
  datasetCount,
  pending,
  failure,
  onRetry,
}: {
  isBuilt: boolean
  datasetCount: number
  pending: string | null
  failure: string | null
  onRetry: () => void
}) {
  if (pending) {
    return (
      <div className="banner banner-busy" style={{ marginTop: 20 }} aria-live="polite">
        <span className="banner-led" />
        <div className="banner-body">
          <div className="banner-title">
            {pending} — the assistant keeps using the previous index meanwhile
          </div>
          <div className="banner-line">
            This is one long request and the service reports no progress while it runs.
            Leave the tab open; it finishes when it finishes.
          </div>
        </div>
      </div>
    )
  }

  if (failure) {
    return (
      <div style={{ marginTop: 20 }}>
        <ErrorPanel
          title="The last attempt failed — the previous index is still serving"
          detail={failure}
          copyText={failure}
          actions={
            <button type="button" className="btn btn-primary" onClick={onRetry}>
              Rebuild again
            </button>
          }
        />
      </div>
    )
  }

  if (!isBuilt) {
    return (
      <div className="banner banner-idle" style={{ marginTop: 20 }}>
        <span className="banner-led" />
        <div className="banner-body">
          <div className="banner-title">No index is built yet</div>
          <div className="banner-line">
            Retrieval cannot answer until the corpus has been indexed. Press{' '}
            <strong>Rebuild index</strong> when the settings below look right.
          </div>
        </div>
        <span className="tag tag-neutral">not built</span>
      </div>
    )
  }

  return (
    <div className="banner banner-good" style={{ marginTop: 20 }}>
      <span className="banner-led" />
      <div className="banner-body">
        <div className="banner-title">Index is built and answering</div>
        <div className="banner-line mono">
          {pluralise(datasetCount, 'corpus dataset')} indexed
        </div>
      </div>
      <span className="tag tag-accent-2">healthy</span>
    </div>
  )
}

/** Corpus datasets Langfuse advertises, plus anything the pipeline is already
 * built from that has since left the listing — dropping it silently would hide
 * part of the live configuration. */
function buildCorpusChoices(available: string[], active: string[]): Choice[] {
  const seen = new Set(available)
  const extras = active.filter((name) => !seen.has(name))
  return [...available, ...extras].sort().map((name) => ({
    value: name,
    label: (
      <span className="mono" style={{ fontSize: 12.5, fontWeight: 600 }}>
        {name}
      </span>
    ),
    note: extras.includes(name)
      ? 'In use, but Langfuse no longer lists it under the corpus folder.'
      : undefined,
  }))
}

function ModelPicker({
  id,
  value,
  options,
  disabled,
  onChange,
}: {
  id: string
  value: string
  options: string[]
  disabled?: boolean
  onChange: (value: string) => void
}) {
  // The upstream listing does not always contain a model the service is
  // already configured with, so the current value is always offered.
  const all = options.includes(value) ? options : [value, ...options]
  if (options.length === 0) {
    return (
      <input
        id={id}
        className="input mono"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    )
  }
  return (
    <select
      id={id}
      className="input mono"
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
    >
      {all.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  )
}

function NumberField({
  label,
  value,
  problem,
  disabled,
  onChange,
}: {
  label: string
  value: string
  problem?: string
  disabled?: boolean
  onChange: (value: string) => void
}) {
  return (
    <Field label={label} hint={problem ? <Bad>{problem}</Bad> : undefined}>
      {(id) => (
        <input
          id={id}
          className={`input mono${problem ? ' input-invalid' : ''}`}
          inputMode="numeric"
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
        />
      )}
    </Field>
  )
}

function TextField({
  label,
  value,
  problem,
  disabled,
  onChange,
}: {
  label: string
  value: string
  problem?: string
  disabled?: boolean
  onChange: (value: string) => void
}) {
  return (
    <Field label={label} hint={problem ? <Bad>{problem}</Bad> : undefined}>
      {(id) => (
        <input
          id={id}
          className={`input mono${problem ? ' input-invalid' : ''}`}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
        />
      )}
    </Field>
  )
}

function Bad({ children }: { children: React.ReactNode }) {
  return <span style={{ color: 'var(--color-alarm-ink)' }}>{children}</span>
}
