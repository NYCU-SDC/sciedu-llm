import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The console is served from the same origin as the API in production, so
// `VITE_API_BASE_URL` defaults to "" and every call is a bare /admin/... path.
// In development that path is proxied to a backend on :8080; point
// `VITE_DEV_API_TARGET` elsewhere if the service runs on another port.
const DEV_API_TARGET = process.env.VITE_DEV_API_TARGET ?? 'http://localhost:8080'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Every /admin call answers promptly now that index rebuilds are scheduled
      // rather than awaited, so this needs no timeout of its own.
      '/admin': {
        target: DEV_API_TARGET,
        changeOrigin: true,
      },
      // /agents is not under /admin, so it needs a rule of its own. It answers
      // Server-Sent Events, which http-proxy forwards chunk by chunk; the
      // timeouts only need to outlast a run, not a rebuild.
      '/agents': {
        target: DEV_API_TARGET,
        changeOrigin: true,
        timeout: 10 * 60 * 1000,
        proxyTimeout: 10 * 60 * 1000,
      },
      // The top bar polls /healthz every few seconds; it is not under /admin
      // either, and a short leash is right for it — an unanswered probe is
      // exactly what the indicator is there to report.
      '/healthz': {
        target: DEV_API_TARGET,
        changeOrigin: true,
      },
    },
  },
})
