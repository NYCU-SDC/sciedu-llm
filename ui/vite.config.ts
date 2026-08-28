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
      '/admin': {
        target: DEV_API_TARGET,
        changeOrigin: true,
        // Rebuilds and resets are long synchronous calls; the proxy must not
        // cut them off before the client's own timeout does.
        timeout: 30 * 60 * 1000,
        proxyTimeout: 30 * 60 * 1000,
      },
    },
  },
})
