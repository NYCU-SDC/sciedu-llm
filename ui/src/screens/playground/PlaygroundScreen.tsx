/* The playground: one conversation with `POST /agents`, rendered the way the
 * protocol means it to be read.
 *
 * Nothing here persists. The transcript lives in component state, the session id
 * is minted once per page load, and neither is written to storage of any kind —
 * a reload is a clean slate on purpose, because this screen exists to try things
 * out, not to keep them. The backend is stateless too: every turn re-sends the
 * whole conversation (see `historyMessages`). */

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Brain, FlaskConical, RotateCcw, Search, Send, Square, Wrench } from 'lucide-react'

import { streamAgents, type AgentEvent, type CastCharacter } from '../../api/agentsStream'
import { langfuseSessionUrl } from '../../api/client'
import { errorMessage } from '../../api/errors'
import { usePresets } from '../../api/hooks'
import { CopyButton } from '../../components/CopyButton'
import { ErrorPanel, QueryError } from '../../components/ErrorPanel'
import { Field, Panel } from '../../components/Panel'
import { PageHeader } from '../../components/States'
import {
  blockIsVisible,
  buildTurn,
  hasInternal,
  historyMessages,
  type SpeakerBlock,
  type StreamedPart,
  type Turn,
} from './transcript'

/** Every run from this screen is traced under one user id, so a Langfuse filter
 * separates manual pokes at the service from anything real. */
const PLAYGROUND_USER = 'playground'

/** The server's own fallback (`AGENTS_DEFAULT_PRESET`). Naming it in the picker
 * costs no extra request — choosing it simply sends no `preset` at all. */
const SERVER_DEFAULT_LABEL = 'server default (default-agents)'

/** `crypto.randomUUID` only exists in a secure context, and this console is
 * often reached over plain http on a lab network — the same reason CopyButton
 * keeps a fallback. The id only has to be unique enough to group traces. */
function newSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  const bytes = new Uint8Array(16)
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    crypto.getRandomValues(bytes)
  } else {
    for (let at = 0; at < bytes.length; at++) bytes[at] = Math.floor(Math.random() * 256)
  }
  const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export function PlaygroundScreen() {
  const presets = usePresets()

  const [sessionId, setSessionId] = useState(newSessionId)
  const [preset, setPreset] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState('')
  const [runningId, setRunningId] = useState<string | null>(null)

  const abort = useRef<AbortController | null>(null)
  const nextId = useRef(0)
  const foot = useRef<HTMLDivElement | null>(null)

  const busy = runningId !== null

  // Abort whatever is in flight when the screen goes away, so a half-read
  // stream does not go on calling setState.
  useEffect(() => () => abort.current?.abort(), [])

  const patch = useCallback((id: string, change: (turn: Turn) => Turn) => {
    setTurns((previous) => previous.map((turn) => (turn.id === id ? change(turn) : turn)))
  }, [])

  const send = () => {
    const content = draft.trim()
    if (!content || busy) return

    const userId = `t${nextId.current++}`
    const assistantId = `t${nextId.current++}`
    const chosen = preset === '' ? undefined : preset
    // The whole conversation so far, plus this message: /agents keeps nothing.
    const messages = [...historyMessages(turns), { role: 'user' as const, content }]

    setTurns((previous) => [
      ...previous,
      { kind: 'user', id: userId, content },
      {
        kind: 'assistant',
        id: assistantId,
        preset: chosen ?? null,
        events: [],
        aborted: false,
        failure: null,
      },
    ])
    setDraft('')
    setRunningId(assistantId)

    const controller = new AbortController()
    abort.current = controller

    void streamAgents(
      { messages, preset: chosen, session: sessionId, user: PLAYGROUND_USER },
      {
        signal: controller.signal,
        onEvent: (event: AgentEvent) => {
          patch(assistantId, (turn) =>
            turn.kind === 'assistant' ? { ...turn, events: [...turn.events, event] } : turn,
          )
        },
      },
    )
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          patch(assistantId, (turn) =>
            turn.kind === 'assistant' ? { ...turn, aborted: true } : turn,
          )
          return
        }
        patch(assistantId, (turn) =>
          turn.kind === 'assistant'
            ? {
                ...turn,
                failure: {
                  title: 'The service did not run this turn',
                  detail: errorMessage(error),
                },
              }
            : turn,
        )
      })
      .finally(() => {
        if (abort.current === controller) abort.current = null
        setRunningId((current) => (current === assistantId ? null : current))
      })
  }

  const stop = () => abort.current?.abort()

  const newSession = () => {
    abort.current?.abort()
    setSessionId(newSessionId())
    setTurns([])
    setDraft('')
  }

  // Follow the conversation as turns are added — but not on every token, which
  // would fight anyone scrolling back through a long run.
  useEffect(() => {
    foot.current?.scrollIntoView({ block: 'nearest' })
  }, [turns.length])

  const nextMessages = useMemo(() => historyMessages(turns), [turns])
  const sessionUrl = langfuseSessionUrl(sessionId)

  return (
    <>
      <PageHeader
        kicker={
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <FlaskConical size={12} strokeWidth={2.75} aria-hidden />
            Manual testing
          </span>
        }
        title="Playground"
        lede="Talk to POST /agents the way a client would, and watch the typed part protocol arrive: who is speaking, what they looked up, and what the machinery underneath was doing."
        actions={
          <button type="button" className="btn btn-secondary" onClick={newSession}>
            <RotateCcw size={14} strokeWidth={2.75} aria-hidden />
            New session
          </button>
        }
      />

      <div className="pg-identity">
        <span className="note">Session</span>
        <span className="mono pg-session">{sessionId}</span>
        <CopyButton text={sessionId} label="Copy id" className="btn btn-ghost pg-mini" />
        <span className="note">
          · traced as user <span className="mono">{PLAYGROUND_USER}</span>
        </span>
        {sessionUrl && (
          <a href={sessionUrl} target="_blank" rel="noreferrer" style={{ fontSize: 12.5 }}>
            open this session in Langfuse
          </a>
        )}
      </div>

      <div className="split split-narrow" style={{ marginTop: 18 }}>
        <div className="col">
          <div className="pg-transcript">
            {turns.length === 0 && (
              <p className="quiet">
                Nothing sent yet. Pick a preset on the right and ask something — a
                multi-character preset will introduce its cast before anybody speaks.
              </p>
            )}
            {turns.map((turn) =>
              turn.kind === 'user' ? (
                <UserBubble key={turn.id} content={turn.content} />
              ) : (
                <AssistantTurnView
                  key={turn.id}
                  events={turn.events}
                  aborted={turn.aborted}
                  failure={turn.failure}
                  preset={turn.preset}
                />
              ),
            )}
            <div ref={foot} />
          </div>

          <Composer
            value={draft}
            busy={busy}
            onChange={setDraft}
            onSend={send}
            onStop={stop}
          />
        </div>

        <aside className="sticky-side">
          <Panel title="Preset">
            <Field
              label="Which behaviour to run"
              hint={
                preset === ''
                  ? 'No preset is sent, so AGENTS_DEFAULT_PRESET decides.'
                  : (
                      <>
                        Sent as <span className="mono">preset</span> on every turn of this
                        session.
                      </>
                    )
              }
            >
              {(id) => (
                <PresetPicker
                  id={id}
                  value={preset}
                  names={presets.data?.map((entry) => entry.name) ?? []}
                  disabled={busy}
                  onChange={setPreset}
                />
              )}
            </Field>
            {presets.isError && (
              <div style={{ marginTop: 12 }}>
                <QueryError
                  what="Could not list the presets"
                  error={presets.error}
                  actions={
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => void presets.refetch()}
                    >
                      Try again
                    </button>
                  }
                />
              </div>
            )}
          </Panel>

          <Panel title="What the next turn sends">
            <div className="kv-list" style={{ fontSize: 12.5 }}>
              <div>
                <span className="note" style={{ width: 66 }}>
                  session
                </span>
                <span className="mono" style={{ fontSize: 11.5, wordBreak: 'break-all' }}>
                  {sessionId}
                </span>
              </div>
              <div>
                <span className="note" style={{ width: 66 }}>
                  user
                </span>
                <span className="mono" style={{ fontSize: 11.5 }}>
                  {PLAYGROUND_USER}
                </span>
              </div>
              <div>
                <span className="note" style={{ width: 66 }}>
                  preset
                </span>
                <span className="mono" style={{ fontSize: 11.5 }}>
                  {preset === '' ? '(not sent)' : preset}
                </span>
              </div>
              <div>
                <span className="note" style={{ width: 66 }}>
                  messages
                </span>
                <span className="mono" style={{ fontSize: 11.5 }}>
                  {nextMessages.length} + your new one
                </span>
              </div>
            </div>
            <details style={{ marginTop: 12 }}>
              <summary>Show the folded conversation</summary>
              <pre className="pg-code" style={{ marginTop: 8 }}>
                {JSON.stringify(nextMessages, null, 2)}
              </pre>
            </details>
            <p className="note" style={{ marginTop: 10 }}>
              /agents remembers nothing between turns, so each request carries the whole
              conversation. A finished answer folds into one assistant message: the text
              its speakers actually said, without the reasoning or the tool traffic.
            </p>
          </Panel>

          <Panel title="Not persisted">
            <p className="note">
              This screen keeps the transcript in the page and nowhere else. Reloading, or
              pressing <strong>New session</strong>, mints a fresh session id and starts
              over.
            </p>
          </Panel>
        </aside>
      </div>
    </>
  )
}

// ── the composer ──────────────────────────────────────────────────────────

function Composer({
  value,
  busy,
  onChange,
  onSend,
  onStop,
}: {
  value: string
  busy: boolean
  onChange: (value: string) => void
  onSend: () => void
  onStop: () => void
}) {
  return (
    <div className="panel pg-composer">
      <textarea
        className="input pg-input"
        placeholder={busy ? 'Waiting for the run to finish…' : 'Ask something. Enter sends, Shift+Enter starts a new line.'}
        value={value}
        disabled={busy}
        rows={3}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            onSend()
          }
        }}
      />
      <div className="pg-composer-actions">
        <span className="note">
          {busy ? 'Streaming — the composer is closed until this turn ends.' : 'Enter sends · Shift+Enter for a new line'}
        </span>
        {busy ? (
          <button type="button" className="btn btn-secondary" onClick={onStop}>
            <Square size={14} strokeWidth={2.75} aria-hidden />
            Stop
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary"
            disabled={value.trim() === ''}
            onClick={onSend}
          >
            <Send size={14} strokeWidth={2.75} aria-hidden />
            Send
          </button>
        )}
      </div>
    </div>
  )
}

function PresetPicker({
  id,
  value,
  names,
  disabled,
  onChange,
}: {
  id: string
  value: string
  names: string[]
  disabled: boolean
  onChange: (value: string) => void
}) {
  // With no listing to choose from — the service is down, or /admin/presets
  // failed — the name becomes a free-text field rather than an empty dropdown,
  // the same fallback the retrieval screen's model picker uses. An unknown name
  // is a 400 from /agents, which the transcript renders like any other refusal.
  if (names.length === 0) {
    return (
      <input
        id={id}
        className="input mono"
        placeholder="preset name (blank = server default)"
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
      <option value="">{SERVER_DEFAULT_LABEL}</option>
      {names.map((name) => (
        <option key={name} value={name}>
          {name}
        </option>
      ))}
    </select>
  )
}

// ── the transcript ────────────────────────────────────────────────────────

function UserBubble({ content }: { content: string }) {
  return (
    <div className="pg-turn pg-turn-user">
      <div className="pg-speaker-head">
        <span className="pg-avatar pg-avatar-user">Y</span>
        <span className="pg-name">You</span>
      </div>
      <p className="pg-text">{content}</p>
    </div>
  )
}

function AssistantTurnView({
  events,
  aborted,
  failure,
  preset,
}: {
  events: AgentEvent[]
  aborted: boolean
  failure: { title: string; detail: string } | null
  preset: string | null
}) {
  const [showInternal, setShowInternal] = useState(false)
  const turn = useMemo(
    () => buildTurn(events, { aborted, failed: failure !== null }),
    [events, aborted, failure],
  )

  const cast = new Map(turn.cast.map((character) => [character.id, character]))
  // The cast is ordered by the server with the orchestrator first, which is the
  // only signal for who is running the show.
  const orchestrator = turn.cast[0]?.id ?? null
  const internalPresent = turn.blocks.some(hasInternal)

  return (
    <div className="pg-turn">
      {turn.cast.length > 0 && (
        <div className="pg-cast">
          {turn.cast.map((character) => (
            <span className="pg-cast-chip" key={character.id}>
              <Avatar character={character} orchestrator={orchestrator} />
              <span className="pg-name">{character.displayName}</span>
              <span className="tag tag-neutral">{character.role}</span>
            </span>
          ))}
          <span className="note" style={{ marginLeft: 'auto' }}>
            cast · {preset ?? 'server default'}
          </span>
        </div>
      )}

      {turn.blocks.map((block) =>
        blockIsVisible(block, showInternal) ? (
          <SpeakerBlockView
            key={block.key}
            block={block}
            cast={cast}
            orchestrator={orchestrator}
            activeKey={turn.activeBlockKey}
            showInternal={showInternal}
          />
        ) : null,
      )}

      {turn.error && (
        <div style={{ marginTop: 10 }}>
          <ErrorPanel
            title="The run failed"
            detail={turn.error.message}
            copyText={`${turn.error.code} — ${turn.error.message}`}
          >
            <p className="alarm-body" style={{ marginTop: 6 }}>
              code <span className="mono">{turn.error.code}</span>
            </p>
          </ErrorPanel>
        </div>
      )}

      {failure && (
        <div style={{ marginTop: 10 }}>
          <ErrorPanel title={failure.title} detail={failure.detail} copyText={failure.detail} />
        </div>
      )}

      <div className="pg-turn-foot">
        {turn.status === 'streaming' && <span className="note">streaming…</span>}
        {turn.status === 'completed' && (
          <span className="note">
            finished · <span className="mono">{turn.finishReason ?? 'unspecified'}</span>
          </span>
        )}
        {turn.status === 'aborted' && <span className="tag tag-outline">stopped</span>}
        {internalPresent && (
          <label className="radio pg-toggle">
            <input
              type="checkbox"
              checked={showInternal}
              onChange={(event) => setShowInternal(event.target.checked)}
            />
            <span className="dot" />
            <span>show internal machinery</span>
          </label>
        )}
      </div>
    </div>
  )
}

function Avatar({
  character,
  orchestrator,
}: {
  character: CastCharacter
  orchestrator: string | null
}) {
  const lead = orchestrator === null || character.id === orchestrator
  return (
    <span
      className={lead ? 'pg-avatar' : 'pg-avatar pg-avatar-2'}
      aria-hidden
      title={character.displayName}
    >
      {[...character.displayName][0] ?? '?'}
    </span>
  )
}

function SpeakerBlockView({
  block,
  cast,
  orchestrator,
  activeKey,
  showInternal,
}: {
  block: SpeakerBlock
  cast: Map<string, CastCharacter>
  orchestrator: string | null
  activeKey: string | null
  showInternal: boolean
}) {
  // A single-character run sends no cast at all, so the agent id ("assistant")
  // is the only name there is.
  const character: CastCharacter = cast.get(block.agent) ?? {
    id: block.agent,
    displayName: block.agent,
    role: '',
  }
  const summoner = block.parent ? (cast.get(block.parent)?.displayName ?? block.parent) : null
  const speaking = block.key === activeKey

  return (
    <div className={block.depth > 0 ? 'pg-speaker pg-speaker-nested' : 'pg-speaker'}>
      <div className="pg-speaker-head">
        <Avatar character={character} orchestrator={orchestrator} />
        <span className="pg-name">{character.displayName}</span>
        {character.role && <span className="tag tag-neutral">{character.role}</span>}
        {summoner && (
          <span className="note">
            summoned by {summoner}
            {block.summonedBy && (
              <>
                {' · '}
                <span className="mono">{block.summonedBy}</span>
              </>
            )}
          </span>
        )}
        {speaking && <span className="pg-live" aria-label="speaking" />}
      </div>

      <div className="pg-speaker-body">
        {block.items.map((item) => {
          if (item.kind === 'part') {
            if (!showInternal && item.part.internal === true) return null
            return <PartView key={item.part.id} part={item.part} showInternal={showInternal} />
          }
          if (!blockIsVisible(item.block, showInternal)) return null
          return (
            <SpeakerBlockView
              key={item.block.key}
              block={item.block}
              cast={cast}
              orchestrator={orchestrator}
              activeKey={activeKey}
              showInternal={showInternal}
            />
          )
        })}
      </div>
    </div>
  )
}

/** What a part shows depends on whether it has finished: until `part_end`
 * lands, the only content there is is what the `delta`s have accumulated. */
function partText(part: StreamedPart): string {
  if (part.done) return part.text ?? ''
  return part.streamed
}

function partArguments(part: StreamedPart): string {
  if (part.done && part.arguments !== undefined) {
    return JSON.stringify(part.arguments, null, 2)
  }
  // Mid-stream a tool call's arguments are raw JSON fragments from the model,
  // which are shown as they arrive rather than held back.
  return part.streamed
}

function PartView({ part, showInternal }: { part: StreamedPart; showInternal: boolean }) {
  const internal = part.internal === true
  const wrap = (children: ReactNode) => (
    <div className={internal ? 'pg-part pg-part-internal' : 'pg-part'}>
      {internal && showInternal && <span className="tag tag-outline pg-internal-tag">internal</span>}
      {children}
    </div>
  )

  if (part.type === 'text') {
    return wrap(
      <p className="pg-text">
        {partText(part)}
        {!part.done && <span className="pg-caret" />}
      </p>,
    )
  }

  if (part.type === 'reasoning') {
    return wrap(
      <details className="pg-aside">
        <summary>
          <Brain size={13} strokeWidth={2.75} aria-hidden /> Reasoning
          {!part.done && ' — still thinking'}
        </summary>
        <p className="pg-aside-body">{partText(part)}</p>
      </details>,
    )
  }

  if (part.type === 'tool_call') {
    const rag = part.name === 'rag_search'
    return wrap(
      <details className="pg-chip">
        <summary>
          {rag ? (
            <Search size={13} strokeWidth={2.75} aria-hidden />
          ) : (
            <Wrench size={13} strokeWidth={2.75} aria-hidden />
          )}
          <span className="mono">{part.name ?? 'tool'}</span>
          <span className={part.done ? 'tag tag-neutral' : 'tag tag-accent'}>
            {part.done ? 'called' : 'calling…'}
          </span>
        </summary>
        <pre className="pg-code">{partArguments(part) || '(no arguments)'}</pre>
      </details>,
    )
  }

  return wrap(
    <details className="pg-chip">
      <summary>
        <Wrench size={13} strokeWidth={2.75} aria-hidden />
        <span className="mono">result</span>
        <span
          className={
            part.status === 'error'
              ? 'tag tag-outline'
              : part.done
                ? 'tag tag-accent-2'
                : 'tag tag-accent'
          }
        >
          {part.done ? (part.status ?? 'returned') : 'running…'}
        </span>
        {part.code && <span className="mono note">{part.code}</span>}
      </summary>
      <pre className="pg-code">{part.content ?? (part.done ? '(no content)' : '…')}</pre>
    </details>,
  )
}
