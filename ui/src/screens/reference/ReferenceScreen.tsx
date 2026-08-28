import { useDatasets, useJudgePrompts, useModels } from '../../api/hooks'
import type { ModelDefaults } from '../../api/types'
import { QueryError } from '../../components/ErrorPanel'
import { Panel } from '../../components/Panel'
import { Loading, PageHeader } from '../../components/States'

export function ReferenceScreen() {
  const models = useModels()
  const datasets = useDatasets()
  const prompts = useJudgePrompts()

  return (
    <>
      <PageHeader
        kicker="Read-only"
        title="What's available"
        lede="Everything the service can currently see. Nothing here is editable — it comes from the model server and from Langfuse."
      />

      <div className="split" style={{ gridTemplateColumns: '1fr 1fr', gap: 14, marginTop: 20 }}>
        <div className="col">
          <Panel title="Models on the server">
            {models.isError ? (
              <QueryError what="Could not list the models" error={models.error} />
            ) : !models.data ? (
              <Loading what="the models" />
            ) : models.data.models.length === 0 ? (
              <p className="quiet">The upstream server advertises no models.</p>
            ) : (
              <>
                <table className="table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Chat</th>
                      <th>Role</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.data?.models.map((id) => {
                      const allowed = models.data.allowed_models.includes(id)
                      return (
                        <tr key={id}>
                          <td className="mono" style={{ fontSize: 12.5 }}>
                            {id}
                          </td>
                          <td style={{ fontSize: 12.5 }}>
                            {allowed ? (
                              <span className="tag tag-accent-2">allowed</span>
                            ) : (
                              <span className="tag tag-neutral">not on allowlist</span>
                            )}
                          </td>
                          <td style={{ fontSize: 12.5, color: 'var(--color-neutral-700)' }}>
                            {roleHints(id, models.data.defaults).join(' · ') || '—'}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
                <p className="note" style={{ marginTop: 12 }}>
                  The allowlist governs <span className="mono">/chat</span> only. An
                  evaluation or an embedding may use any model the server advertises.
                </p>
              </>
            )}
          </Panel>

          <Panel title="Scoring prompts">
            {prompts.isError ? (
              <QueryError what="Could not list the judge prompts" error={prompts.error} />
            ) : !prompts.data ? (
              <Loading what="the judge prompts" />
            ) : prompts.data.length === 0 ? (
              <p className="quiet">
                Langfuse has no prompts under the judge folder yet.
              </p>
            ) : (
              <table className="table">
                <tbody>
                  {prompts.data?.map((prompt) => (
                    <tr key={prompt.name}>
                      <td className="mono" style={{ fontSize: 12.5 }}>
                        {prompt.name}
                      </td>
                      <td style={{ fontSize: 12.5, color: 'var(--color-neutral-700)' }}>
                        {prompt.label}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <p className="note" style={{ marginTop: 12 }}>
              The wording of each prompt lives in Langfuse; the service only fetches it by
              name.
            </p>
          </Panel>
        </div>

        <Panel title="Langfuse datasets">
          {datasets.isError ? (
            <QueryError what="Could not list the Langfuse datasets" error={datasets.error} />
          ) : !datasets.data ? (
            <Loading what="the datasets" />
          ) : datasets.data.corpus.length + datasets.data.questions.length === 0 ? (
            <p className="quiet">
              Langfuse lists nothing under the corpus or questions folders.
            </p>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Group</th>
                </tr>
              </thead>
              <tbody>
                {datasets.data?.corpus.map((dataset) => (
                  <tr key={dataset.name}>
                    <td className="mono" style={{ fontSize: 12.5 }}>
                      {dataset.name}
                    </td>
                    <td>
                      <span className="tag tag-accent">corpus</span>
                    </td>
                  </tr>
                ))}
                {datasets.data?.questions.map((dataset) => (
                  <tr key={dataset.name}>
                    <td className="mono" style={{ fontSize: 12.5 }}>
                      {dataset.name}
                    </td>
                    <td>
                      <span className="tag tag-accent-2">questions</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="note" style={{ marginTop: 12 }}>
            Add a dataset in Langfuse under <span className="mono">corpus/</span> or{' '}
            <span className="mono">questions/</span> and it appears here after a refresh.
            The service lists only those two folders — the{' '}
            <span className="mono">config/presets</span> dataset the presets live in is not
            part of this listing, and item counts are not reported.
          </p>
        </Panel>
      </div>
    </>
  )
}

/** What the service would reach for this model by default. Only the four
 * defaults `GET /admin/models` actually returns. */
function roleHints(id: string, defaults: ModelDefaults): string[] {
  const hints: string[] = []
  if (defaults.eval_model === id) hints.push('default for evaluations')
  if (defaults.judge_model === id && defaults.judge_model !== defaults.eval_model) {
    hints.push('default for scoring')
  }
  if (defaults.embedding_model === id) hints.push('embedding (current)')
  if (defaults.rerank_model === id) hints.push('rerank (current)')
  return hints
}
