import { useMemo, useState, type ReactNode } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

/** One selectable dataset.
 *
 * `name` is the full Langfuse name ("corpus/ver3/biology") and is always what
 * selection reports back; everything on screen is derived from it. */
export interface DatasetItem {
    name: string;
    /** The name with its top-level group prefix stripped — what `/admin/datasets`
     * already sends as `label`. Derived from `name` when absent. */
    label?: string;
    note?: ReactNode;
}

interface Leaf {
    name: string;
    /** What the row shows: the path inside its folder. */
    text: string;
    note?: ReactNode;
}

interface Folder {
    path: string;
    leaves: Leaf[];
}

/** "corpus/ver3/biology" → "ver3/biology"; "corpus/foo" → "foo". */
function stripGroup(name: string): string {
    const at = name.indexOf("/");
    return at === -1 ? name : name.slice(at + 1);
}

function byText(a: Leaf, b: Leaf): number {
    return a.text.localeCompare(b.text);
}

/** One level of nesting: the first folder segment of the display path becomes a
 * group, anything deeper stays joined with "/" in the leaf's own label, and a
 * bare "corpus/foo" is a leaf at the root. */
function organise(items: DatasetItem[]): { folders: Folder[]; loose: Leaf[] } {
    const byFolder = new Map<string, Leaf[]>();
    const loose: Leaf[] = [];

    for (const item of items) {
        const display = item.label ?? stripGroup(item.name);
        const at = display.indexOf("/");
        if (at === -1) {
            loose.push({ name: item.name, text: display, note: item.note });
            continue;
        }
        const folder = display.slice(0, at);
        const leaves = byFolder.get(folder) ?? [];
        leaves.push({
            name: item.name,
            text: display.slice(at + 1),
            note: item.note,
        });
        byFolder.set(folder, leaves);
    }

    const folders = [...byFolder.entries()]
        .map(([path, leaves]) => ({ path, leaves: [...leaves].sort(byText) }))
        .sort((a, b) => a.path.localeCompare(b.path));
    return { folders, loose: [...loose].sort(byText) };
}

/** The dataset picker both the retrieval screen and the evaluations form use.
 *
 * Langfuse names are paths, so they are shown as one: the group prefix comes
 * off, the next segment becomes a collapsible folder with its own
 * select-the-lot checkbox, and the whole set has one above it. Only the display
 * is folded — every value handed back to the caller is the full name. */
export function FolderDatasetPicker({
    items,
    selected,
    onChange,
    disabled,
    boxed,
    empty = "沒有可選項目。",
}: {
    items: DatasetItem[];
    selected: string[];
    onChange: (next: string[]) => void;
    disabled?: boolean;
    /** Constrain to a scrolling box — for the denser pickers on the evals form. */
    boxed?: boolean;
    empty?: ReactNode;
}) {
    const { folders, loose } = useMemo(() => organise(items), [items]);
    const [collapsed, setCollapsed] = useState<string[]>([]);

    const chosen = useMemo(() => new Set(selected), [selected]);

    if (items.length === 0) {
        return <p className="note">{empty}</p>;
    }

    const apply = (names: string[], next: boolean) => {
        const updated = new Set(selected);
        for (const name of names) {
            if (next) updated.add(name);
            else updated.delete(name);
        }
        onChange([...updated].sort());
    };

    const countOn = (names: string[]) =>
        names.filter((name) => chosen.has(name)).length;

    const everyName = items.map((item) => item.name);
    const onCount = countOn(everyName);

    const leafRow = (leaf: Leaf) => (
        <TriCheck
            key={leaf.name}
            checked={chosen.has(leaf.name)}
            disabled={disabled}
            title={leaf.name}
            onChange={(next) => apply([leaf.name], next)}
        >
            <span className="mono picker-leaf">{leaf.text}</span>
            {leaf.note && <span className="picker-note">{leaf.note}</span>}
        </TriCheck>
    );

    return (
        <div className="picker">
            <div className="picker-head">
                <TriCheck
                    checked={onCount === everyName.length}
                    indeterminate={onCount > 0 && onCount < everyName.length}
                    disabled={disabled}
                    bare
                    onChange={(next) => apply(everyName, next)}
                >
                    <span className="picker-all">全選</span>
                </TriCheck>
                <span className="picker-count mono">
                    已選取 {onCount} / {everyName.length}
                </span>
            </div>

            <div className={boxed ? "picker-body picker-boxed" : "picker-body"}>
                {folders.map((folder) => {
                    const names = folder.leaves.map((leaf) => leaf.name);
                    const on = countOn(names);
                    const open = !collapsed.includes(folder.path);
                    return (
                        <div className="picker-folder" key={folder.path}>
                            <div className="picker-folder-head">
                                <button
                                    type="button"
                                    className="picker-toggle"
                                    aria-expanded={open}
                                    aria-label={`${open ? "收合" : "展開"} ${folder.path}`}
                                    onClick={() =>
                                        setCollapsed((previous) =>
                                            previous.includes(folder.path)
                                                ? previous.filter(
                                                      (path) =>
                                                          path !== folder.path
                                                  )
                                                : [...previous, folder.path]
                                        )
                                    }
                                >
                                    {open ? (
                                        <ChevronDown
                                            size={14}
                                            strokeWidth={2.75}
                                            aria-hidden
                                        />
                                    ) : (
                                        <ChevronRight
                                            size={14}
                                            strokeWidth={2.75}
                                            aria-hidden
                                        />
                                    )}
                                </button>
                                <TriCheck
                                    checked={on === names.length}
                                    indeterminate={on > 0 && on < names.length}
                                    disabled={disabled}
                                    bare
                                    onChange={(next) => apply(names, next)}
                                >
                                    <span className="picker-folder-name">
                                        {folder.path}
                                    </span>
                                </TriCheck>
                                <span className="picker-count mono">
                                    {on}/{names.length}
                                </span>
                            </div>
                            {open && (
                                <div className="picker-leaves">
                                    {folder.leaves.map(leafRow)}
                                </div>
                            )}
                        </div>
                    );
                })}
                {loose.map(leafRow)}
            </div>
        </div>
    );
}

/** The design system's `.radio` + `.dot` pair over a native checkbox, with the
 * third state a folder needs. `indeterminate` is a DOM property rather than an
 * attribute, so it is set on the node itself. */
function TriCheck({
    checked,
    indeterminate,
    disabled,
    title,
    bare,
    onChange,
    children,
}: {
    checked: boolean;
    indeterminate?: boolean;
    disabled?: boolean;
    title?: string;
    /** Without the tinted row background — for the two header checkboxes. */
    bare?: boolean;
    onChange: (next: boolean) => void;
    children: ReactNode;
}) {
    return (
        <label
            className={bare ? "radio picker-bare" : "radio check-row"}
            title={title}
        >
            <input
                type="checkbox"
                checked={checked}
                disabled={disabled}
                ref={(node) => {
                    if (node) node.indeterminate = indeterminate ?? false;
                }}
                onChange={(event) => onChange(event.target.checked)}
            />
            <span className="dot" />
            <span style={{ flex: 1, minWidth: 0 }}>{children}</span>
        </label>
    );
}
