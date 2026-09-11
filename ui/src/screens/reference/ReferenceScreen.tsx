import { useJudgePrompts, useModels } from "../../api/hooks";
import type { ModelInfo } from "../../api/types";
import { QueryError } from "../../components/ErrorPanel";
import { Panel } from "../../components/Panel";
import { Loading, PageHeader } from "../../components/States";
import {
    groupModelsByMode,
    isChatModelMode,
    isModelAllowed,
} from "./modelGroups";

export function ReferenceScreen() {
    const models = useModels();
    const prompts = useJudgePrompts();
    const modelGroups = groupModelsByMode(models.data?.models ?? []);

    return (
        <>
            <PageHeader
                kicker="系統概覽"
                title="可用資源"
                lede="服務目前可存取的模型與評分提示詞。此處無法編輯；資料來自模型伺服器與 Langfuse。"
            />

            <div className="col" style={{ marginTop: 20 }}>
                {models.isError ? (
                    <Panel title="伺服器上的模型">
                        <QueryError what="無法列出模型" error={models.error} />
                    </Panel>
                ) : !models.data ? (
                    <Panel title="伺服器上的模型">
                        <Loading what="模型" />
                    </Panel>
                ) : modelGroups.length === 0 ? (
                    <Panel title="伺服器上的模型">
                        <p className="quiet">上游伺服器未提供任何模型。</p>
                    </Panel>
                ) : (
                    <>
                        {modelGroups.map((group) => (
                            <ModelPanel
                                key={group.mode}
                                title={group.title}
                                mode={group.mode}
                                models={group.models}
                                allowedModels={models.data.allowed_models}
                            />
                        ))}
                        <p className="note">
                            允許清單只影響 <span className="mono">/chat</span>
                            ；未設定時，所有上游模型均視為允許。其他功能不受允許清單限制。
                        </p>
                    </>
                )}

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
                                {prompts.data.map((prompt) => (
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
                        每個提示詞的內容儲存在 Langfuse；服務只依名稱取得它。
                    </p>
                </Panel>
            </div>
        </>
    );
}

function ModelPanel({
    title,
    mode,
    models,
    allowedModels,
}: {
    title: string;
    mode: string;
    models: ModelInfo[];
    allowedModels: string[];
}) {
    const showAvailability = isChatModelMode(mode);

    return (
        <Panel title={title}>
            <table className="table">
                <thead>
                    <tr>
                        <th>模型</th>
                        {showAvailability && <th>可用狀態</th>}
                    </tr>
                </thead>
                <tbody>
                    {models.map((model) => (
                        <tr key={model.id}>
                            <td className="mono" style={{ fontSize: 12.5 }}>
                                {model.id}
                            </td>
                            {showAvailability && (
                                <td style={{ fontSize: 12.5 }}>
                                    {isModelAllowed(model.id, allowedModels) ? (
                                        <span className="tag tag-accent-2">
                                            可用
                                        </span>
                                    ) : (
                                        <span className="tag tag-neutral">
                                            不在允許清單中
                                        </span>
                                    )}
                                </td>
                            )}
                        </tr>
                    ))}
                </tbody>
            </table>
        </Panel>
    );
}
