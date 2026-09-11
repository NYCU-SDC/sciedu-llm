import type { CSSProperties, ReactNode } from "react";

/** The design-system checkbox.
 *
 * The native input remains in the document for keyboard and screen-reader
 * behavior; `.dot` is the consistently rendered control. Keeping the label,
 * input and visual mark together prevents a screen from accidentally falling
 * back to the browser's platform-specific checkbox.
 */
export function Checkbox({
    checked,
    onChange,
    children,
    disabled,
    indeterminate,
    title,
    className,
    style,
}: {
    checked: boolean;
    onChange: (checked: boolean) => void;
    children: ReactNode;
    disabled?: boolean;
    indeterminate?: boolean;
    title?: string;
    className?: string;
    style?: CSSProperties;
}) {
    return (
        <label
            className={`radio checkbox${className ? ` ${className}` : ""}`}
            title={title}
            style={style}
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
            <span className="dot" aria-hidden />
            {children}
        </label>
    );
}
