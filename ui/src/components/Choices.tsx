import type { ReactNode } from "react";

export interface Choice {
    value: string;
    label: ReactNode;
    note?: ReactNode;
    trailing?: ReactNode;
}

/** A checkbox list built on the design system's `.radio` + `.dot` pair with a
 * squared dot — the mockup's course-material picker. Native checkboxes stay
 * underneath, so keyboard and screen-reader behaviour is the browser's. */
export function CheckList({
    choices,
    selected,
    onToggle,
    disabled,
    boxed,
    empty = "沒有可選項目。",
}: {
    choices: Choice[];
    selected: string[];
    onToggle: (value: string, next: boolean) => void;
    disabled?: boolean;
    /** Constrain to a scrolling box — for the denser pickers on the evals form. */
    boxed?: boolean;
    empty?: ReactNode;
}) {
    if (choices.length === 0) {
        return <p className="note">{empty}</p>;
    }
    const rows = choices.map((choice) => {
        const on = selected.includes(choice.value);
        return (
            <label className="radio check-row" key={choice.value}>
                <input
                    type="checkbox"
                    checked={on}
                    disabled={disabled}
                    onChange={(event) =>
                        onToggle(choice.value, event.target.checked)
                    }
                />
                <span className="dot" />
                <span style={{ flex: 1, minWidth: 0 }}>
                    {choice.label}
                    {choice.note && (
                        <span
                            style={{
                                display: "block",
                                fontSize: 12,
                                color: "var(--color-neutral-600)",
                            }}
                        >
                            {choice.note}
                        </span>
                    )}
                </span>
                {choice.trailing}
            </label>
        );
    });
    return boxed ? (
        <div className="choice-box">{rows}</div>
    ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {rows}
        </div>
    );
}

/** A stack of radio rows, styled like the mockup's course-material choice. */
export function RadioList<T extends string>({
    name,
    options,
    value,
    onChange,
    disabled,
}: {
    name: string;
    options: { value: T; label: ReactNode; note?: ReactNode }[];
    value: T;
    onChange: (value: T) => void;
    disabled?: boolean;
}) {
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {options.map((option) => (
                <label className="radio radio-row" key={option.value}>
                    <input
                        type="radio"
                        name={name}
                        checked={value === option.value}
                        disabled={disabled}
                        onChange={() => onChange(option.value)}
                    />
                    <span className="dot" style={{ marginRight: 8 }} />
                    <span style={{ flex: 1 }}>
                        {option.label}
                        {option.note && (
                            <span
                                className="note"
                                style={{ display: "block", marginTop: 2 }}
                            >
                                {option.note}
                            </span>
                        )}
                    </span>
                </label>
            ))}
        </div>
    );
}
