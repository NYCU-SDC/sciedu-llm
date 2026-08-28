import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Plus, Trash2 } from 'lucide-react'

import { ApiError } from '../../api/client'
import { errorMessage, locPath } from '../../api/errors'
import { useModels, usePreset, usePresetMutations, useTools } from '../../api/hooks'
import type { Preset, PresetCharacter, PresetDetail, ToolChoice } from '../../api/types'
import { MAX_STEPS_CAP } from '../../api/types'
import { RadioList } from '../../components/Choices'
import { ConfirmDialog } from '../../components/ConfirmDialog'
import { ErrorPanel, QueryError } from '../../components/ErrorPanel'
import { Field, Panel } from '../../components/Panel'
import { Loading, PageHeader } from '../../components/States'
import {
  RAG_SEARCH,
  SUMMON_SUBAGENT,
  blankCharacter,
  blankPreset,
  checkPresetShape,
  courseMaterialMode,
  forcedRagObjections,
  formatTools,
  labelForLoc,
  normalisePreset,
  parseTools,
  setCourseMaterialMode,
  type CourseMaterialMode,
  type ShapeProblem,
} from './presetShape'

type Tab = 'form' | 'json'

const TOOL_CHOICES: { value: ToolChoice; label: string }[] = [
  { value: 'auto', label: 'auto — the model decides' },
  { value: 'required', label: 'required — must use a tool first' },
  { value: 'none', label: 'none — no tools' },
]

const COURSE_MATERIAL: { value: CourseMaterialMode; label: React.ReactNode }[] = [
  { value: 'never', label: 'Never search — the model answers from its own knowledge' },
  { value: 'always', label: 'Always search before answering' },
  {
    value: 'decide',
    label: (
      <>
        Let the model decide, using the <span className="mono">{RAG_SEARCH}</span> tool
      </>
    ),
  },
]

export function PresetEditorScreen() {
  const { name } = useParams<{ name: string }>()
  const isNew = name === undefined
  const navigate = useNavigate()

  const loaded = usePreset(name)
  const models = useModels()
  const tools = useTools()
  const { save, remove } = usePresetMutations()

  const [tab, setTab] = useState<Tab>('form')
  const [jsonText, setJsonText] = useState('')
  const [localProblems, setLocalProblems] = useState<ShapeProblem[] | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const detail: PresetDetail | undefined = loaded.data

  // The buffer only exists once the user edits something: until then the screen
  // renders the served document directly. Deriving it this way rather than
  // copying it into state on arrival means a background refetch can never
  // clobber an edit in progress, and needs no effect.
  const [draft, setDraft] = useState<Preset | null>(null)
  const fresh = useMemo(() => blankPreset(), [])
  const base = isNew ? fresh : (detail?.definition ?? null)
  const preset = draft ?? base

  const setPreset = (next: Preset) => setDraft(next)
  const editPreset = (fn: (previous: Preset) => Preset) =>
    setDraft((previous) => {
      const from = previous ?? base
      return from ? fn(from) : previous
    })

  const deletable = detail ? !detail.builtin || detail.shadowed_builtin : false

  const toJson = (value: Preset) => JSON.stringify(value, null, 2)

  const switchTab = (next: Tab) => {
    if (next === tab) return
    if (next === 'json' && preset) {
      setJsonText(toJson(preset))
      setLocalProblems(null)
      setTab('json')
      return
    }
    // Going back to the form needs a document the form can bind to.
    const parsed = parseJson(jsonText)
    if (!parsed.ok) {
      setLocalProblems(parsed.problems)
      return
    }
    setPreset(parsed.preset)
    setLocalProblems(null)
    setTab('form')
  }

  /** Local JSON checking only — the semantic rules run on the server at save
   * time, and that is the validation that counts. */
  const validate = () => {
    const parsed = parseJson(jsonText)
    setLocalProblems(parsed.ok ? [] : parsed.problems)
    if (parsed.ok) setPreset(parsed.preset)
  }

  const currentDocument = (): Preset | null => {
    if (tab === 'form') return preset
    const parsed = parseJson(jsonText)
    if (!parsed.ok) {
      setLocalProblems(parsed.problems)
      return null
    }
    setLocalProblems(null)
    setPreset(parsed.preset)
    return parsed.preset
  }

  const onSave = () => {
    const document = currentDocument()
    if (!document) return
    if (!document.name) {
      setLocalProblems([{ path: 'name', message: 'A preset needs a name before it can be saved.' }])
      return
    }
    save.mutate(
      { name: document.name, preset: document },
      {
        onSuccess: (saved) => {
          if (isNew || saved.name !== name) {
            void navigate(`/presets/${encodeURIComponent(saved.name)}`, { replace: true })
          }
        },
      },
    )
  }

  if (!isNew && loaded.isError) {
    return (
      <>
        <PageHeader title={name ?? 'Preset'} back={<BackLink />} mono />
        <div style={{ marginTop: 20 }}>
          <QueryError what={`Could not open the preset '${name}'`} error={loaded.error} />
        </div>
      </>
    )
  }

  if (!preset) {
    return (
      <>
        <PageHeader title={name ?? 'Preset'} back={<BackLink />} mono />
        <Loading what="the preset" />
      </>
    )
  }

  const update = (patch: Partial<Preset>) => editPreset((previous) => ({ ...previous, ...patch }))

  const updateCharacter = (index: number, patch: Partial<PresetCharacter>) =>
    editPreset((previous) => {
      const characters = previous.characters.map((character, at) =>
        at === index ? { ...character, ...patch } : character,
      )
      // Keep `orchestrator` pointing at the same character when its id changes.
      const wasOrchestrator = previous.characters[index]?.id === previous.orchestrator
      const orchestrator =
        wasOrchestrator && patch.id !== undefined ? patch.id : previous.orchestrator
      return { ...previous, characters, orchestrator }
    })

  const addCharacter = () =>
    editPreset((previous) => {
      if (previous.characters.length >= 2) return previous
      const extra = blankCharacter()
      const characters = [...previous.characters, extra]
      // A second character is only reachable if the orchestrator may summon it.
      const withSummon = characters.map((character) =>
        character.id === previous.orchestrator &&
        !character.tools.includes(SUMMON_SUBAGENT)
          ? { ...character, tools: [...character.tools, SUMMON_SUBAGENT] }
          : character,
      )
      return { ...previous, characters: withSummon }
    })

  const removeCharacter = (index: number) =>
    editPreset((previous) => {
      const characters = previous.characters.filter((_, at) => at !== index)
      // With nobody left to summon, the tool is invalid — the backend rejects it.
      const cleaned = characters.map((character) => ({
        ...character,
        tools: character.tools.filter((tool) => tool !== SUMMON_SUBAGENT),
      }))
      return { ...previous, characters: cleaned }
    })

  const objections = forcedRagObjections(preset)
  const allowed = models.data?.allowed_models ?? []
  const renamed = !isNew && detail && preset.name !== detail.name

  return (
    <>
      <PageHeader
        back={<BackLink />}
        title={preset.name || (isNew ? 'New preset' : (name ?? ''))}
        mono
        lede={<StoredIn isNew={isNew} detail={detail} />}
        actions={
          <>
            <div className="seg">
              {(['form', 'json'] as Tab[]).map((option) => (
                <label className="seg-opt" key={option}>
                  <input
                    type="radio"
                    name="preset-tab"
                    checked={tab === option}
                    onChange={() => switchTab(option)}
                  />
                  <span>{option === 'form' ? 'Form' : 'JSON'}</span>
                </label>
              ))}
            </div>
            {tab === 'json' && (
              <button type="button" className="btn btn-secondary" onClick={validate}>
                Validate
              </button>
            )}
            <button
              type="button"
              className="btn btn-primary"
              onClick={onSave}
              disabled={save.isPending}
            >
              {save.isPending ? 'Saving…' : 'Save & reload registry'}
            </button>
          </>
        }
      />

      {localProblems !== null && localProblems.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <ErrorPanel
            title={`This document is not the right shape — ${localProblems.length} problem${localProblems.length === 1 ? '' : 's'}`}
          >
            <div className="alarm-list">
              {localProblems.map((problem) => (
                <div className="mono" style={{ fontSize: 12.5 }} key={problem.path + problem.message}>
                  {problem.path} — {problem.message}
                </div>
              ))}
            </div>
            <p className="alarm-body">
              Checked here in the browser. The service runs the full validation when you
              save.
            </p>
          </ErrorPanel>
        </div>
      )}

      {localProblems !== null && localProblems.length === 0 && (
        <div className="banner banner-good" style={{ marginTop: 20 }}>
          <span className="banner-led" />
          <div className="banner-body">
            <div className="banner-title">The document has the right shape</div>
            <div className="banner-line">
              Tool names, and the rules about summoned characters and forced retrieval, are
              checked by the service when you save.
            </div>
          </div>
        </div>
      )}

      {save.error && (
        <div style={{ marginTop: 14 }}>
          <SaveError error={save.error} preset={preset} />
        </div>
      )}

      {remove.error && (
        <div style={{ marginTop: 14 }}>
          <QueryError what="Could not delete this preset" error={remove.error} />
        </div>
      )}

      <div className="split split-narrow" style={{ marginTop: 14 }}>
        {tab === 'json' ? (
          <Panel title="The preset document">
            <p className="note" style={{ marginBottom: 12 }}>
              This is exactly what is stored as the <span className="mono">config/presets</span>{' '}
              dataset item, and exactly what{' '}
              <span className="mono">PUT /admin/presets/{preset.name || '{name}'}</span>{' '}
              accepts.
            </p>
            <textarea
              className="input code"
              spellCheck={false}
              value={jsonText}
              onChange={(event) => {
                setJsonText(event.target.value)
                setLocalProblems(null)
              }}
            />
          </Panel>
        ) : (
          <div className="col">
            <Panel title="Basics">
              <div className="grid-2">
                <Field
                  label="Preset name"
                  hint={
                    renamed
                      ? `Saving under a new name creates a second preset; '${detail?.name}' stays as it is.`
                      : 'Lowercase letters, digits, "-" and "_".'
                  }
                >
                  {(id) => (
                    <input
                      id={id}
                      className="input mono"
                      value={preset.name}
                      onChange={(event) => update({ name: event.target.value })}
                    />
                  )}
                </Field>
                <Field label="Model" hint="The service's own default is used when this is left unset.">
                  {(id) => (
                    <select
                      id={id}
                      className="input mono"
                      value={preset.model ?? ''}
                      onChange={(event) =>
                        update({ model: event.target.value === '' ? null : event.target.value })
                      }
                    >
                      <option value="">— the server's default model —</option>
                      {modelOptions(allowed, preset.model).map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  )}
                </Field>
              </div>
              <Field label="What this preset is for" style={{ marginTop: 14 }}>
                {(id) => (
                  <input
                    id={id}
                    className="input"
                    value={preset.description}
                    onChange={(event) => update({ description: event.target.value })}
                  />
                )}
              </Field>
            </Panel>

            <Panel title="Cast">
              <p className="note" style={{ marginBottom: 14 }}>
                One assistant answers by default. You can add a second character the first
                one may call in when it needs a different voice.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {preset.characters.map((character, index) => (
                  <CharacterCard
                    key={index}
                    character={character}
                    isOrchestrator={character.id === preset.orchestrator}
                    onChange={(patch) => updateCharacter(index, patch)}
                    onRemove={
                      preset.characters.length > 1 && character.id !== preset.orchestrator
                        ? () => removeCharacter(index)
                        : undefined
                    }
                  />
                ))}
                {preset.characters.length < 2 && (
                  <button
                    type="button"
                    className="btn btn-ghost"
                    style={{ alignSelf: 'flex-start' }}
                    onClick={addCharacter}
                  >
                    <Plus size={15} strokeWidth={2.75} aria-hidden />
                    Add a summonable character
                  </button>
                )}
              </div>
            </Panel>

            <Panel title="Course material">
              <RadioList
                name="course-material"
                options={COURSE_MATERIAL}
                value={courseMaterialMode(preset)}
                onChange={(mode) => setPreset(setCourseMaterialMode(preset, mode))}
              />
              {objections.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <ErrorPanel title="The service will refuse this combination">
                    <div className="alarm-list">
                      {objections.map((objection) => (
                        <div style={{ fontSize: 12.5 }} key={objection}>
                          {objection}
                        </div>
                      ))}
                    </div>
                  </ErrorPanel>
                </div>
              )}
            </Panel>

            <Panel title="Limits">
              <div className="grid-2">
                <Field
                  label="Maximum steps per reply"
                  hint={`How many times it may use a tool before it must answer. At most ${MAX_STEPS_CAP}.`}
                >
                  {(id) => (
                    <input
                      id={id}
                      className="input mono"
                      inputMode="numeric"
                      value={preset.max_steps}
                      onChange={(event) =>
                        update({ max_steps: toInt(event.target.value, preset.max_steps) })
                      }
                    />
                  )}
                </Field>
                <Field label="Tool choice">
                  {(id) => (
                    <select
                      id={id}
                      className="input"
                      value={preset.tool_choice}
                      onChange={(event) =>
                        update({ tool_choice: event.target.value as ToolChoice })
                      }
                    >
                      {TOOL_CHOICES.map((choice) => (
                        <option key={choice.value} value={choice.value}>
                          {choice.label}
                        </option>
                      ))}
                    </select>
                  )}
                </Field>
              </div>
            </Panel>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 16,
                flexWrap: 'wrap',
                padding: 2,
              }}
            >
              <button
                type="button"
                className="btn btn-primary"
                style={{ fontSize: 14.5, padding: '11px 22px' }}
                onClick={onSave}
                disabled={save.isPending}
              >
                {save.isPending ? 'Saving…' : 'Save & reload registry'}
              </button>
              {deletable && (
                <>
                  <button
                    type="button"
                    className="btn btn-ghost btn-danger"
                    onClick={() => setConfirmDelete(true)}
                    disabled={remove.isPending}
                  >
                    <Trash2 size={15} strokeWidth={2.75} aria-hidden />
                    Delete this preset
                  </button>
                  {detail?.shadowed_builtin && (
                    <p className="note" style={{ maxWidth: '38ch' }}>
                      Deleting brings the built-in <span className="mono">{detail.name}</span>{' '}
                      back.
                    </p>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        <aside className="sticky-side">
          <Panel title="Tools you can name">
            {!tools.data ? (
              <Loading what="the tools" />
            ) : (
              <div className="kv-list">
                {(tools.data ?? []).map((tool) => (
                  <div key={tool.name}>
                    <span className="mono" style={{ fontSize: 12.5, flex: 'none', fontWeight: 600 }}>
                      {tool.name}
                    </span>
                    <span
                      style={{
                        fontSize: 11.5,
                        color: 'var(--color-neutral-600)',
                        textAlign: 'right',
                        flex: 1,
                      }}
                    >
                      {tool.description}
                    </span>
                  </div>
                ))}
              </div>
            )}
            <p className="note" style={{ marginTop: 10 }}>
              <span className="mono">{SUMMON_SUBAGENT}</span> only works on the orchestrator,
              and only when there is a second character to summon.
            </p>
          </Panel>

          <Panel title="Models on the allowlist">
            {models.isError ? (
              <p className="note">Unavailable — {errorMessage(models.error)}</p>
            ) : !models.data ? (
              <Loading what="the models" />
            ) : allowed.length === 0 ? (
              <p className="note">
                No allowlist is configured, so any model the upstream server advertises may
                be used.
              </p>
            ) : (
              allowed.map((model) => (
                <div className="mono" style={{ fontSize: 12.5, padding: '3px 0' }} key={model}>
                  {model}
                </div>
              ))
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
            Prefer to write the document directly? Switch to <strong>JSON</strong> above —
            the service runs the same validation either way.
          </div>
        </aside>
      </div>

      {confirmDelete && detail && (
        <ConfirmDialog
          title={`Delete '${detail.name}'?`}
          danger
          body={
            detail.shadowed_builtin ? (
              <>
                The Langfuse item is removed and the built-in{' '}
                <span className="mono">{detail.name}</span> goes back into service.
              </>
            ) : (
              <>
                The Langfuse item is removed and the preset stops being served. Anything
                calling it by name will get an error.
              </>
            )
          }
          confirmLabel="Delete it"
          busy={remove.isPending}
          onCancel={() => setConfirmDelete(false)}
          onConfirm={() => {
            setConfirmDelete(false)
            remove.mutate(detail.name, { onSuccess: () => void navigate('/presets') })
          }}
        />
      )}
    </>
  )
}

function BackLink() {
  return (
    <Link to="/presets" style={{ fontSize: 12.5 }}>
      ← All presets
    </Link>
  )
}

function StoredIn({ isNew, detail }: { isNew: boolean; detail: PresetDetail | undefined }) {
  if (isNew) {
    return (
      <span style={{ fontSize: 13 }}>
        Not saved yet · saving writes it to the Langfuse dataset{' '}
        <span className="mono">config/presets</span>
      </span>
    )
  }
  if (!detail) return null
  if (detail.builtin && !detail.shadowed_builtin) {
    return (
      <span style={{ fontSize: 13 }}>
        Built into the service · saving stores a Langfuse copy that shadows it, and the
        built-in stays available underneath
      </span>
    )
  }
  if (detail.builtin) {
    return (
      <span style={{ fontSize: 13 }}>
        Stored in Langfuse · shadows the built-in <span className="mono">{detail.name}</span>
      </span>
    )
  }
  return <span style={{ fontSize: 13 }}>Stored in Langfuse</span>
}

function CharacterCard({
  character,
  isOrchestrator,
  onChange,
  onRemove,
}: {
  character: PresetCharacter
  isOrchestrator: boolean
  onChange: (patch: Partial<PresetCharacter>) => void
  onRemove?: () => void
}) {
  return (
    <div className="cast-card">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span className={isOrchestrator ? 'tag tag-neutral' : 'tag tag-accent'}>
          {isOrchestrator ? 'orchestrator' : 'summonable'}
        </span>
        <span className="mono" style={{ fontSize: 13, fontWeight: 600 }}>
          {character.id || '(no id)'}
        </span>
        {onRemove && (
          <button
            type="button"
            className="btn btn-ghost btn-danger"
            style={{ marginLeft: 'auto' }}
            onClick={onRemove}
          >
            Remove
          </button>
        )}
      </div>
      <div className="grid-2" style={{ marginTop: 12 }}>
        <Field label="Id (used in the document)">
          {(id) => (
            <input
              id={id}
              className="input mono"
              value={character.id}
              onChange={(event) => onChange({ id: event.target.value })}
            />
          )}
        </Field>
        <Field label="Display name (shown to the student)">
          {(id) => (
            <input
              id={id}
              className="input"
              value={character.display_name}
              onChange={(event) => onChange({ display_name: event.target.value })}
            />
          )}
        </Field>
      </div>
      <div className="grid-2" style={{ marginTop: 12 }}>
        <Field label="Role">
          {(id) => (
            <input
              id={id}
              className="input mono"
              value={character.role}
              onChange={(event) => onChange({ role: event.target.value })}
            />
          )}
        </Field>
        <Field
          label="Prompt (Langfuse)"
          hint={isOrchestrator ? undefined : 'A summoned character needs a prompt of its own.'}
        >
          {(id) => (
            <input
              id={id}
              className="input mono"
              placeholder={isOrchestrator ? 'optional' : 'required'}
              value={character.prompt_name ?? ''}
              onChange={(event) =>
                onChange({ prompt_name: event.target.value === '' ? null : event.target.value })
              }
            />
          )}
        </Field>
      </div>
      <div className="grid-2" style={{ marginTop: 12 }}>
        <Field label="Tools it may call" hint="Comma-separated names from the list on the right.">
          {(id) => (
            <input
              id={id}
              className="input mono"
              value={formatTools(character.tools)}
              onChange={(event) => onChange({ tools: parseTools(event.target.value) })}
            />
          )}
        </Field>
        {!isOrchestrator && (
          <Field label="Steps when summoned" hint={`At most ${MAX_STEPS_CAP}.`}>
            {(id) => (
              <input
                id={id}
                className="input mono"
                inputMode="numeric"
                value={character.max_steps}
                onChange={(event) =>
                  onChange({ max_steps: toInt(event.target.value, character.max_steps) })
                }
              />
            )}
          </Field>
        )}
      </div>
    </div>
  )
}

/** A 422 from `PUT /admin/presets/{name}` arrives either as a flat string (the
 * name-mismatch check) or as pydantic's list of problems. The list gets mapped
 * back onto the form's own wording where the `loc` allows it. */
function SaveError({ error, preset }: { error: unknown; preset: Preset }) {
  const problems = error instanceof ApiError ? error.problems : []
  const status = error instanceof ApiError ? error.status : null

  if (problems.length === 0) {
    return (
      <ErrorPanel
        title={
          status === 502
            ? 'Langfuse rejected the write'
            : status
              ? `The service rejected this document — ${status}`
              : 'The service could not be reached'
        }
        detail={errorMessage(error)}
        copyText={errorMessage(error)}
      />
    )
  }

  return (
    <ErrorPanel
      title={`The service rejected this document — ${problems.length} problem${problems.length === 1 ? '' : 's'}`}
      copyText={problems.map((p) => `${locPath(p.loc)}: ${p.msg}`).join('\n')}
    >
      <div className="alarm-list">
        {problems.map((problem, index) => (
          <div className="mono" style={{ fontSize: 12.5 }} key={index}>
            <strong>{labelForLoc(problem.loc, preset)}</strong> — {problem.msg}
          </div>
        ))}
      </div>
    </ErrorPanel>
  )
}

type ParseResult =
  | { ok: true; preset: Preset }
  | { ok: false; problems: ShapeProblem[] }

function parseJson(text: string): ParseResult {
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch (error) {
    return {
      ok: false,
      problems: [
        { path: '(root)', message: `not valid JSON — ${errorMessage(error)}` },
      ],
    }
  }
  const problems = checkPresetShape(value)
  if (problems.length > 0) return { ok: false, problems }
  return { ok: true, preset: normalisePreset(value as Record<string, unknown>) }
}

function modelOptions(allowed: string[], current: string | null): string[] {
  if (current && !allowed.includes(current)) return [current, ...allowed]
  return allowed
}

function toInt(text: string, fallback: number): number {
  const value = Number(text)
  return /^\d+$/.test(text.trim()) && Number.isFinite(value) ? value : fallback
}
