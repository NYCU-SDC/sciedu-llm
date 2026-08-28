/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where the admin API lives. Empty (the default) means same-origin, which is
   * how the console is deployed; the dev server proxies /admin instead. */
  readonly VITE_API_BASE_URL?: string
  /** The Langfuse base URL, e.g. https://langfuse.example.org. Every "open in
   * Langfuse" link is hidden when this is unset rather than guessed at. */
  readonly VITE_LANGFUSE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
