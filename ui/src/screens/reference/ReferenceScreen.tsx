import { useDatasets, useJudgePrompts, useModels } from "../../api/hooks";
import type { ModelDefaults } from "../../api/types";
import { QueryError } from "../../components/ErrorPanel";
import { Panel } from "../../components/Panel";
import { Loading, PageHeader } from "../../components/States";

export function ReferenceScreen() {
    const models = useModels();
    const datasets = useDatasets();
    const prompts = useJudgePrompts();

    return (
        <>
            <PageHeader
                kicker="系統概覽"
                title="可用資源"
                lede="服務目前可存取的一切資源。此處無法編輯；資料來自模型伺服器與 Langfuse。"
            />

            <div
                className="split"
                style={{
                    gridTemplateColumns: "1fr 1fr",
                    gap: 14,
                    marginTop: 20,
                }}
            >
                <div className="col">
                    <Panel title="伺服器上的模型">
                        {models.isError ? (
                            <QueryError
                                what="無法列出模型"
                                error={models.error}
                            />
                        ) : !models.data ? (
                            <Loading what="模型" />
                        ) : models.data.models.length === 0 ? (
                            <p className="quiet">上游伺服器未提供任何模型。</p>
                        ) : (
                            <>
                                <table className="table">
                                    <thead>
                                        <tr>
                                            <th>模型</th>
                                            <th>聊天</th>
                                            <th>用途</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {models.data?.models.map((id) => {
                                            const allowed =
                                                models.data.allowed_models.includes(
                                                    id
                                                );
                                            return (
                                                <tr key={id}>
                                                    <td
                                                        className="mono"
                                                        style={{
                                                            fontSize: 12.5,
                                                        }}
                                                    >
                                                        {id}
                                                    </td>
                                                    <td
                                                        style={{
                                                            fontSize: 12.5,
                                                        }}
                                                    >
                                                        {allowed ? (
                                                            <span className="tag tag-accent-2">
                                                                允許
                                                            </span>
                                                        ) : (
                                                            <span className="tag tag-neutral">
                                                                不在允許清單中
                                                            </span>
                                                        )}
                                                    </td>
                                                    <td
                                                        style={{
                                                            fontSize: 12.5,
                                                            color: "var(--color-neutral-700)",
                                                        }}
                                                    >
                                                        {roleHints(
                                                            id,
                                                            models.data.defaults
                                                        ).join(" · ") || "—"}
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                                <p className="note" style={{ marginTop: 12 }}>
                                    允許清單只影響{" "}
                                    <span className="mono">/chat</span>
                                    。評估或嵌入可使用伺服器提供的任何模型。
                                </p>
                            </>
                        )}
                    </Panel>

                    <Panel title="評分提示詞">
                        {prompts.isError ? (
                            <QueryError
                                what="無法列出評分提示詞"
                                error={prompts.error}
                            />
                        ) : !prompts.data ? (
                            <Loading what="評分提示詞" />
                        ) : prompts.data.length === 0 ? (
                            <p className="quiet">
                                Langfuse 的 judge 資料夾目前沒有提示詞。
                            </p>
                        ) : (
                            <table className="table">
                                <tbody>
                                    {prompts.data?.map((prompt) => (
                                        <tr key={prompt.name}>
                                            <td
                                                className="mono"
                                                style={{ fontSize: 12.5 }}
                                            >
                                                {prompt.name}
                                            </td>
                                            <td
                                                style={{
                                                    fontSize: 12.5,
                                                    color: "var(--color-neutral-700)",
                                                }}
                                            >
                                                {prompt.label}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        )}
                        <p className="note" style={{ marginTop: 12 }}>
                            每個提示詞的內容儲存在
                            Langfuse；服務只依名稱取得它。
                        </p>
                    </Panel>
                </div>

                <Panel title="Langfuse 資料集">
                    {datasets.isError ? (
                        <QueryError
                            what="無法列出 Langfuse 資料集"
                            error={datasets.error}
                        />
                    ) : !datasets.data ? (
                        <Loading what="資料集" />
                    ) : datasets.data.corpus.length +
                          datasets.data.questions.length ===
                      0 ? (
                        <p className="quiet">
                            Langfuse 在 corpus 或 questions
                            資料夾下沒有任何資料。
                        </p>
                    ) : (
                        <table className="table">
                            <thead>
                                <tr>
                                    <th>資料集</th>
                                    <th>群組</th>
                                </tr>
                            </thead>
                            <tbody>
                                {datasets.data?.corpus.map((dataset) => (
                                    <tr key={dataset.name}>
                                        <td
                                            className="mono"
                                            style={{ fontSize: 12.5 }}
                                        >
                                            {dataset.name}
                                        </td>
                                        <td>
                                            <span className="tag tag-accent">
                                                corpus
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                                {datasets.data?.questions.map((dataset) => (
                                    <tr key={dataset.name}>
                                        <td
                                            className="mono"
                                            style={{ fontSize: 12.5 }}
                                        >
                                            {dataset.name}
                                        </td>
                                        <td>
                                            <span className="tag tag-accent-2">
                                                questions
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                    <p className="note" style={{ marginTop: 12 }}>
                        在 Langfuse 的 <span className="mono">corpus/</span> 或{" "}
                        <span className="mono">questions/</span>{" "}
                        下新增資料集，重新整理後就會顯示於此。
                        服務只列出這兩個資料夾；存放預設值的{" "}
                        <span className="mono">config/presets</span>{" "}
                        資料集不在此清單中，也不會顯示項目數量。
                    </p>
                </Panel>
            </div>
        </>
    );
}

/** What the service would reach for this model by default. Only the four
 * defaults `GET /admin/models` actually returns. */
function roleHints(id: string, defaults: ModelDefaults): string[] {
    const hints: string[] = [];
    if (defaults.eval_model === id) hints.push("評估預設值");
    if (
        defaults.judge_model === id &&
        defaults.judge_model !== defaults.eval_model
    ) {
        hints.push("評分預設值");
    }
    if (defaults.embedding_model === id) hints.push("嵌入（目前）");
    if (defaults.rerank_model === id) hints.push("重排序（目前）");
    return hints;
}
