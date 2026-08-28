import { useMemo, useState } from 'react'

import { ApiError } from '../../api/client'
import { errorMessage } from '../../api/errors'
import { useDatasets, useModels, useRagConfig, useRagMutations } from '../../api/hooks'
import type { NamedResource } from '../../api/types'
import { ConfirmDialog } from '../../components/ConfirmDialog'
import { ErrorPanel, QueryError } from '../../components/ErrorPanel'
import { FolderDatasetPicker, type DatasetItem } from '../../components/FolderDatasetPicker'
import { Field, Panel } from '../../components/Panel'
import { Loading, PageHeader } from '../../components/States'
import { RebuildTag } from '../../components/StatusTag'
import { buildUpdate, diffDraft, toDraft, type NumericKey, type RagDraft } from './draft'

/** The plain-language notes the mockup puts beside each retrieval knob. */
const KNOBS: { key: NumericKey; label: string; note: string }[] = [
  {
    key: 'bm25_top_n',
    label: '關鍵字前 n 筆',
    note: '從精確措辭搜尋（BM25）保留的段落。',
  },
  {
    key: 'dense_top_n',
    label: '語意前 n 筆',
    note: '從語意相近搜尋保留的段落。',
  },
  {
    key: 'rrf_k',
    label: '合併常數（RRF k）',
    note: '兩份結果清單的混合強度；數值越高越平緩。',
  },
  {
    key: 'rerank_pool_size',
    label: '重排序候選集',
    note: '合併後會再次閱讀與重新排序的段落數量。',
  },
  {
    key: 'final_k',
    label: '最終 k',
    note: '助理回答前實際閱讀的段落數量。',
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
              title="此服務未啟用檢索功能"
              detail={errorMessage(config.error)}
            >
              <p className="alarm-body" style={{ marginTop: 10 }}>
                請在服務環境中設定 <span className="mono">RAG_CORPUS_DATASETS</span> 後重新啟動；
                在此之前沒有可設定的項目。
              </p>
            </ErrorPanel>
          ) : (
            <QueryError
              what="無法讀取檢索設定"
              error={config.error}
              actions={
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => void config.refetch()}
                >
                  再試一次
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
        <Loading what="目前的檢索設定" />
      </>
    )
  }

  const live = config.data
  const corpusItems = buildCorpusItems(datasets.data?.corpus ?? [], live.corpus_datasets)
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
              ? '正在套用變更，接著重建索引'
              : '正在套用變更'
            : rebuild.isPending
              ? '正在重建索引'
              : reset.isPending
                ? '正在還原伺服器預設值，接著重建索引'
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
          <Panel title="課程教材">
            <p className="note" style={{ marginBottom: 14 }}>
              助理可引用 <span className="mono">corpus/</span> 下哪些 Langfuse 資料集。
            </p>
            {datasets.isError ? (
              <QueryError
                what="無法列出 Langfuse 資料集"
                error={datasets.error}
                actions={
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => void datasets.refetch()}
                  >
                    再試一次
                  </button>
                }
              />
            ) : !datasets.data ? (
              <Loading what="資料集清單" />
            ) : (
              <FolderDatasetPicker
                items={corpusItems}
                selected={current.corpus_datasets}
                disabled={busy}
                empty="Langfuse 的 corpus 資料夾目前沒有資料集。"
                onChange={(next) => set('corpus_datasets', next)}
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
                讀取自 Langfuse ·{' '}
                <button
                  type="button"
                  className="link-btn"
                  onClick={() => void datasets.refetch()}
                >
                  重新整理清單
                </button>
              </span>
            </div>
          </Panel>

          <Panel title="模型">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div className="grid-2" style={{ alignItems: 'start' }}>
                <div>
                  <Field label="嵌入模型">
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
                    <RebuildTag>會重建索引</RebuildTag>
                  </div>
                </div>
                <p className="note" style={{ paddingTop: 19 }}>
                  將文件轉為數字表示，以找出語意相近的段落。變更後必須重新讀取所有文件。
                </p>
              </div>
              <div className="grid-2" style={{ alignItems: 'start' }}>
                <Field label="重排序模型">
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
                  第二道處理會依段落回答問題的程度重新排序候選段落，可隨時變更。
                </p>
              </div>
            </div>
            {models.isError && (
              <p className="note" style={{ marginTop: 12 }}>
                模型清單無法取得（{errorMessage(models.error)}），目前改為自由輸入欄位。
              </p>
            )}
          </Panel>

          <Panel title="文件切分方式">
            <p className="note" style={{ marginBottom: 14 }}>
              長文件會切成彼此重疊的片段。較小的片段更精確；較大的片段會保留更多上下文。
            </p>
            <div className="grid-2">
              <NumberField
                label="片段大小（字元）"
                value={current.chunk_size}
                problem={problemFor('chunk_size')}
                disabled={busy}
                onChange={(value) => set('chunk_size', value)}
              />
              <NumberField
                label="片段重疊（字元）"
                value={current.chunk_overlap}
                problem={problemFor('chunk_overlap')}
                disabled={busy}
                onChange={(value) => set('chunk_overlap', value)}
              />
            </div>
            <div style={{ marginTop: 12 }}>
              <RebuildTag>變更這些設定會重建索引</RebuildTag>
            </div>
          </Panel>

          <Panel title="段落搜尋方式">
            <p className="note" style={{ marginBottom: 14 }}>
              關鍵字與語意搜尋會並行執行，再將結果合併、篩選與重新排序後交給助理。
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
            <Panel title="回答提示詞">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <TextField
                  label="系統提示詞（Langfuse）"
                  value={current.generator_system_prompt_name}
                  problem={problemFor('generator_system_prompt_name')}
                  disabled={busy}
                  onChange={(value) => set('generator_system_prompt_name', value)}
                />
                <TextField
                  label="使用者提示詞（Langfuse）"
                  value={current.generator_user_prompt_name}
                  problem={problemFor('generator_user_prompt_name')}
                  disabled={busy}
                  onChange={(value) => set('generator_user_prompt_name', value)}
                />
              </div>
              <p className="note" style={{ marginTop: 12 }}>
                提示詞內容儲存在 Langfuse；這裡只指定要取得哪個提示詞。
              </p>
            </Panel>
            <Panel title="處理量">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <NumberField
                  label="嵌入批次大小"
                  value={current.embedding_batch_size}
                  problem={problemFor('embedding_batch_size')}
                  disabled={busy}
                  onChange={(value) => set('embedding_batch_size', value)}
                />
                <NumberField
                  label="最大並行數"
                  value={current.max_concurrency}
                  problem={problemFor('max_concurrency')}
                  disabled={busy}
                  onChange={(value) => set('max_concurrency', value)}
                />
              </div>
              <p className="note" style={{ marginTop: 12 }}>
                此設定決定重建時對模型伺服器的負載。若因速率限制而失敗，請調低數值。
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
              {apply.isPending ? '套用中…' : '套用至執行中的服務'}
            </button>
            {changes.length > 0 && (
              <p className="note" style={{ maxWidth: '46ch' }}>
                {rebuildsOnApply
                  ? `${rebuildCount === 1 ? '有一項變更需要' : `有 ${rebuildCount} 項變更需要`}重建索引，可能需要一些時間。`
                  : '目前的變更不需要重建索引。'}
              </p>
            )}
          </div>
        </div>

        <aside className="panel sticky-aside">
          <h5 className="sect" style={{ margin: 0 }}>
            未儲存的變更
          </h5>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 12 }}>
            {changes.length === 0 ? (
              <p className="note">
                尚未編輯任何設定。畫面顯示的就是服務目前使用的設定。
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
                        <RebuildTag>會重建索引</RebuildTag>
                      </div>
                    )}
                  </div>
                ))}
                <p className="note">按下「套用」前，不會將任何內容傳送至服務。</p>
                <button
                  type="button"
                  className="btn btn-ghost"
                  style={{ alignSelf: 'flex-start' }}
                  disabled={busy}
                  onClick={() => setDraft(null)}
                >
                  捨棄變更
                </button>
              </>
            )}
          </div>
        </aside>
      </div>

      {confirmReset && (
        <ConfirmDialog
          title="要還原為伺服器預設值嗎？"
          body={
            <>
              此程序持有的所有覆寫設定都會移除，並還原服務
              <span className="mono">RAG_*</span> 環境變數的值。之後會立即重建索引，可能需要數分鐘。
            </>
          }
          confirmLabel="還原並重建"
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
      kicker="即時服務"
      title="檢索設定"
      lede="助理回答前如何在課程教材中查找資訊。以下每項設定目前都正在使用。"
      actions={
        <>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={disabled}
            onClick={onReset}
          >
            還原伺服器預設值
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={disabled}
            onClick={onRebuild}
          >
            重建索引
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
            {pending} — 在此期間助理會持續使用先前的索引
          </div>
          <div className="banner-line">
            這是一個耗時的請求，服務執行期間不會回報進度。請保持此分頁開啟，完成後即會結束。
          </div>
        </div>
      </div>
    )
  }

  if (failure) {
    return (
      <div style={{ marginTop: 20 }}>
        <ErrorPanel
          title="上次嘗試失敗，但仍在使用先前的索引"
          detail={failure}
          copyText={failure}
          actions={
            <button type="button" className="btn btn-primary" onClick={onRetry}>
              再次重建
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
          <div className="banner-title">尚未建立索引</div>
          <div className="banner-line">
            語料庫建立索引前無法進行檢索回答。確認下方設定無誤後，請按{' '}
            <strong>重建索引</strong>。
          </div>
        </div>
        <span className="tag tag-neutral">未建立</span>
      </div>
    )
  }

  return (
    <div className="banner banner-good" style={{ marginTop: 20 }}>
      <span className="banner-led" />
      <div className="banner-body">
        <div className="banner-title">索引已建立並可供回答</div>
        <div className="banner-line mono">
          已建立 {datasetCount} 個語料庫資料集的索引
        </div>
      </div>
      <span className="tag tag-accent-2">正常</span>
    </div>
  )
}

/** Corpus datasets Langfuse advertises, plus anything the pipeline is already
 * built from that has since left the listing — dropping it silently would hide
 * part of the live configuration. */
function buildCorpusItems(available: NamedResource[], active: string[]): DatasetItem[] {
  const listed = new Map(available.map((entry) => [entry.name, entry.label]))
  const extras = active.filter((name) => !listed.has(name))
  return [
    ...available.map((entry) => ({ name: entry.name, label: entry.label })),
    ...extras.map((name) => ({
      name,
      note: '正在使用，但 Langfuse 已不再於 corpus 資料夾下列出此項目。',
    })),
  ]
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
