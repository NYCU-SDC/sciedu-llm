import { useEffect, useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { api } from '../../api/client'
import { errorMessage } from '../../api/errors'
import type { Preset, PresetDetail } from '../../api/types'
import { checkPresetShape, normalisePreset, type ShapeProblem } from './presetShape'

/** One document out of the pasted text, checked but not yet sent. */
interface Parsed {
  key: string
  /** What to call it on screen — its name, or its position when it has none. */
  title: string
  /** Null when the shape check found something; nothing invalid is ever sent. */
  preset: Preset | null
  problems: ShapeProblem[]
}

interface Outcome {
  ok: boolean
  message: string
}

/** Paste-in bulk authoring: raw JSON lives here, and nowhere else in the
 * console. Each document is shape-checked in the browser — enough to catch a
 * missing name or a typo before it costs a round trip — and then written with
 * one `PUT /admin/presets/{name}` per preset, in order, so a rejection reports
 * against the document it came from instead of failing the whole paste. The
 * service runs the real validation on every one of those calls. */
export function ImportPresetsDialog({ onClose }: { onClose: () => void }) {
  const client = useQueryClient()
  const [text, setText] = useState('')
  const [results, setResults] = useState<Record<string, Outcome>>({})
  const [busy, setBusy] = useState(false)

  const { error, documents } = useMemo(() => parseDocuments(text), [text])
  const valid = documents.filter((document) => document.preset !== null)
  const done = Object.keys(results).length > 0

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !busy) onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [busy, onClose])

  const run = async () => {
    setBusy(true)
    setResults({})
    for (const entry of documents) {
      const preset = entry.preset
      if (!preset) continue
      try {
        await api.put<PresetDetail>(
          `/admin/presets/${encodeURIComponent(preset.name)}`,
          preset,
        )
        setResults((previous) => ({
          ...previous,
          [entry.key]: { ok: true, message: 'saved' },
        }))
      } catch (failure) {
        setResults((previous) => ({
          ...previous,
          [entry.key]: { ok: false, message: errorMessage(failure) },
        }))
      }
    }
    await client.invalidateQueries({ queryKey: ['presets'] })
    setBusy(false)
  }

  return (
    <div
      className="dialog-backdrop"
      onClick={(event) => {
        if (event.target === event.currentTarget && !busy) onClose()
      }}
    >
      <div
        className="dialog dialog-wide"
        role="dialog"
        aria-modal="true"
        aria-label="Import presets"
      >
        <div className="dialog-title">Import presets</div>
        <p className="note">
          Paste one preset document, or an array of them. Each is written with{' '}
          <span className="mono">PUT /admin/presets/{'{name}'}</span> under its own{' '}
          <span className="mono">name</span>, replacing any preset already stored there.
        </p>

        <textarea
          className="input mono"
          spellCheck={false}
          placeholder='{ "name": "tutor", "orchestrator": "assistant", "characters": [ … ] }'
          style={{ minHeight: 190, maxHeight: '34vh', fontSize: 12.5, lineHeight: 1.6 }}
          value={text}
          disabled={busy}
          onChange={(event) => {
            setText(event.target.value)
            setResults({})
          }}
        />

        {error && (
          <p className="note" style={{ color: 'var(--color-alarm-ink)' }}>
            {error}
          </p>
        )}

        {documents.length > 0 && (
          <div className="import-list">
            {documents.map((entry) => (
              <DocumentRow key={entry.key} entry={entry} outcome={results[entry.key]} />
            ))}
          </div>
        )}

        <div className="dialog-actions">
          <button type="button" className="btn btn-secondary" disabled={busy} onClick={onClose}>
            {done && !busy ? 'Close' : 'Cancel'}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy || valid.length === 0}
            onClick={() => void run()}
          >
            {busy
              ? 'Importing…'
              : valid.length === 0
                ? 'Import'
                : `Import ${valid.length === 1 ? 'this preset' : `these ${valid.length} presets`}`}
          </button>
        </div>
      </div>
    </div>
  )
}

function DocumentRow({ entry, outcome }: { entry: Parsed; outcome: Outcome | undefined }) {
  const state = outcome ? (outcome.ok ? 'saved' : 'rejected') : entry.preset ? 'ready' : 'bad'
  return (
    <div className="import-row">
      <div className="row" style={{ gap: 10 }}>
        <span className="mono" style={{ fontSize: 12.5, fontWeight: 600, flex: 1, minWidth: 0 }}>
          {entry.title}
        </span>
        <span className={`tag ${state === 'saved' ? 'tag-accent-2' : state === 'ready' ? 'tag-neutral' : 'tag-outline'}`}>
          {state === 'saved'
            ? 'saved'
            : state === 'rejected'
              ? 'rejected'
              : state === 'ready'
                ? 'ready'
                : 'not the right shape'}
        </span>
      </div>
      {entry.problems.map((problem) => (
        <div
          className="mono import-problem"
          key={problem.path + problem.message}
        >
          {problem.path} — {problem.message}
        </div>
      ))}
      {outcome && !outcome.ok && (
        <div className="mono import-problem">{outcome.message}</div>
      )}
    </div>
  )
}

/** One document or an array of them, each checked on its own. A parse failure
 * is about the whole paste and is reported as such. */
function parseDocuments(text: string): { error: string | null; documents: Parsed[] } {
  const trimmed = text.trim()
  if (!trimmed) return { error: null, documents: [] }

  let value: unknown
  try {
    value = JSON.parse(trimmed)
  } catch (failure) {
    return { error: `That is not valid JSON — ${errorMessage(failure)}`, documents: [] }
  }

  const list = Array.isArray(value) ? value : [value]
  if (list.length === 0) {
    return { error: 'That array is empty, so there is nothing to import.', documents: [] }
  }

  return {
    error: null,
    documents: list.map((entry, index) => {
      const problems = checkPresetShape(entry)
      const record =
        typeof entry === 'object' && entry !== null && !Array.isArray(entry)
          ? (entry as Record<string, unknown>)
          : {}
      const name = typeof record.name === 'string' ? record.name : ''
      return {
        key: String(index),
        title: name || `document ${index + 1}`,
        preset: problems.length === 0 ? normalisePreset(record) : null,
        problems,
      }
    }),
  }
}
