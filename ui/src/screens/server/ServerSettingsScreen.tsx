import { useSearchParams } from "react-router-dom";
import { PageHeader } from "../../components/States";
import { RagSettingsPanel } from "../rag/RagSettingsPanel";
import { AvailableResourcesPanel } from "../reference/AvailableResourcesPanel";

export function ServerSettingsScreen() {
    const [search, setSearch] = useSearchParams();
    const resources = search.get("section") === "resources";
    return (
        <>
            <PageHeader
                kicker="伺服器"
                title="伺服器設定"
                lede="管理全域 RAG 設定，查看服務可使用的模型與提示詞。"
            />
            <div
                role="tablist"
                aria-label="伺服器設定分類"
                style={{
                    display: "flex",
                    gap: 10,
                    marginTop: 20,
                    marginBottom: 22,
                }}
            >
                <button
                    type="button"
                    role="tab"
                    aria-selected={!resources}
                    aria-controls="server-rag"
                    id="tab-rag"
                    className={`btn ${resources ? "btn-secondary" : "btn-primary"}`}
                    onClick={() => setSearch({})}
                >
                    RAG 設定
                </button>
                <button
                    type="button"
                    role="tab"
                    aria-selected={resources}
                    aria-controls="server-resources"
                    id="tab-resources"
                    className={`btn ${resources ? "btn-primary" : "btn-secondary"}`}
                    onClick={() => setSearch({ section: "resources" })}
                >
                    可用資源
                </button>
            </div>
            <div
                role="tabpanel"
                id="server-resources"
                aria-labelledby="tab-resources"
                hidden={!resources}
            >
                <AvailableResourcesPanel />
            </div>
            <div
                role="tabpanel"
                id="server-rag"
                aria-labelledby="tab-rag"
                hidden={resources}
            >
                <RagSettingsPanel />
            </div>
        </>
    );
}
