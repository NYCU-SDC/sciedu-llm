import { useState } from 'react'
import { useQueries } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'

import { api } from '../../api/client'
import { keys, usePresetMutations, usePresetReport, usePresets } from '../../api/hooks'
import type { Preset, PresetDetail, PresetSummary } from '../../api/types'
import { ErrorPanel, QueryError } from '../../components/ErrorPanel'
import { Loading, PageHeader } from '../../components/States'
import { PresetSourceTag } from '../../components/StatusTag'
import { formatUnixSeconds, pluralise } from '../../lib/format'
import { ImportPresetsDialog } from './ImportPresetsDialog'
import { describeRagMode } from './presetShape'

export function PresetsScreen() {
  const navigate = useNavigate()
  const presets = usePresets()
  const report = usePresetReport()
  const { refresh } = usePresetMutations()
  const [importing, setImporting] = useState(false)

  // `GET /admin/presets` answers names and provenance only; the model, cast,
  // tool count and RAG mode live on the document, so the table fills its
  // remaining columns from one detail request per preset. Those reads are
  // served from the registry's in-memory map — no Langfuse round trip.
  const details = useQueries({
    queries: (presets.data ?? []).map((summary) => ({
      queryKey: keys.preset(summary.name),
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        api.get<PresetDetail>(`/admin/presets/${encodeURIComponent(summary.name)}`, signal),
    })),
  })

  const documents = new Map<string, Preset>()
  ;(presets.data ?? []).forEach((summary, index) => {
    const document = details[index]?.data?.definition
    if (document) documents.set(summary.name, document)
  })

  const errorEntries = Object.entries(report.data?.errors ?? {})

  return (
    <>
      <PageHeader
        kicker="config/presets"
        title="Behaviour presets"
        lede="A preset is one named way for the assistant to behave: which model it uses, which prompts each character speaks from, what tools it may reach for, and whether it searches your course material."
        actions={
          <>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={refresh.isPending}
              onClick={() => refresh.mutate()}
            >
              {refresh.isPending ? 'Reloading…' : 'Reload from Langfuse'}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setImporting(true)}
            >
              Import presets
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => void navigate('/presets/new')}
            >
              New preset
            </button>
          </>
        }
      />

      <LoadBanner
        loading={!report.data && !report.isError}
        loaded={report.data?.loaded.length ?? null}
        fetchedAt={report.data?.fetched_at ?? null}
        rejected={errorEntries.length}
      />

      {report.isError && (
        <div style={{ marginTop: 14 }}>
          <QueryError what="Could not reload the presets from Langfuse" error={report.error} />
        </div>
      )}

      <div className="panel table-wrap" style={{ marginTop: 14 }}>
        {presets.isError ? (
          <QueryError what="Could not list the presets" error={presets.error} />
        ) : !presets.data ? (
          <Loading what="the presets" />
        ) : presets.data.length === 0 ? (
          <p className="quiet">
            No presets are being served. That should not happen — the built-ins are
            code-defined and always available.
          </p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Preset</th>
                <th>Model</th>
                <th>Cast</th>
                <th>Course material</th>
                <th>Tools</th>
                <th>Where it comes from</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {presets.data?.map((summary) => (
                <PresetRow
                  key={summary.name}
                  summary={summary}
                  document={documents.get(summary.name)}
                />
              ))}
            </tbody>
          </table>
        )}
      </div>
      <p className="note" style={{ marginTop: 10 }}>
        Built-in presets ship with the service and can always be restored. A preset stored
        in Langfuse with the same name takes over from the built-in one; delete it and the
        built-in comes back.
      </p>

      {errorEntries.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <ErrorPanel
            title={`${pluralise(errorEntries.length, 'entry', 'entries')} in config/presets could not be loaded`}
            copyText={errorEntries.map(([id, message]) => `item ${id}\n${message}`).join('\n\n')}
          >
            <div className="alarm-body mono">
              {errorEntries.map(([itemId, message]) => (
                <div key={itemId} style={{ marginBottom: 8 }}>
                  <strong>item {itemId}</strong>
                  <br />
                  {message}
                </div>
              ))}
            </div>
            <p className="alarm-body" style={{ marginTop: 4 }}>
              Those items were skipped; everything else stayed in service. Fix them in the
              Langfuse dataset, or open the preset here and re-save it.
            </p>
          </ErrorPanel>
        </div>
      )}

      {importing && <ImportPresetsDialog onClose={() => setImporting(false)} />}
    </>
  )
}

function PresetRow({
  summary,
  document,
}: {
  summary: PresetSummary
  document: Preset | undefined
}) {
  const cast = document
    ? document.characters.length > 1
      ? `orchestrator + ${document.characters.length - 1}`
      : 'single'
    : '…'
  const toolCount = document
    ? document.characters.reduce((total, character) => total + character.tools.length, 0)
    : null

  return (
    <tr>
      <td>
        <Link to={`/presets/${encodeURIComponent(summary.name)}`} className="mono"
          style={{ fontSize: 13, fontWeight: 600 }}>
          {summary.name}
        </Link>
        {summary.description && <span className="cell-sub">{summary.description}</span>}
      </td>
      <td className="mono" style={{ fontSize: 12.5 }}>
        {document ? (document.model ?? 'server default') : '…'}
      </td>
      <td style={{ fontSize: 13 }}>{cast}</td>
      <td style={{ fontSize: 13 }}>{document ? describeRagMode(document) : '…'}</td>
      <td className="mono" style={{ fontSize: 12.5 }}>
        {toolCount === null ? '…' : toolCount === 0 ? '—' : toolCount}
      </td>
      <td>
        <PresetSourceTag builtin={summary.builtin} shadowed={summary.shadowed_builtin} />
      </td>
      <td className="right">
        <Link
          className="btn btn-ghost"
          to={`/presets/${encodeURIComponent(summary.name)}`}
        >
          {summary.builtin && !summary.shadowed_builtin ? 'view' : 'edit'}
        </Link>
      </td>
    </tr>
  )
}

/** What the last registry load produced. The backend has no `GET /load-report`
 * on purpose — `POST /refresh` is the idempotent way to ask, so this line is
 * the result of that call. */
function LoadBanner({
  loading,
  loaded,
  fetchedAt,
  rejected,
}: {
  loading: boolean
  loaded: number | null
  fetchedAt: number | null
  rejected: number
}) {
  if (loading) {
    return (
      <div className="banner banner-idle" style={{ marginTop: 20 }}>
        <span className="banner-led" />
        <div className="banner-body">
          <div className="banner-title">Reading the preset dataset…</div>
        </div>
      </div>
    )
  }
  if (loaded === null) return null

  const failedFetch = fetchedAt === null
  return (
    <div
      className={`banner ${failedFetch ? 'banner-idle' : 'banner-good'}`}
      style={{ marginTop: 20 }}
    >
      <span className="banner-led" />
      <div className="banner-body">
        <div className="banner-title">
          {failedFetch
            ? 'Langfuse could not be read — the presets already in service are still being served'
            : `Last load ${formatUnixSeconds(fetchedAt)}`}
        </div>
        <div className="banner-line">
          <strong>{pluralise(loaded, 'preset')} served</strong>
          {rejected > 0 && `, ${pluralise(rejected, 'entry', 'entries')} rejected`}
        </div>
      </div>
      {rejected > 0 && <span className="tag tag-outline">see below</span>}
    </div>
  )
}
