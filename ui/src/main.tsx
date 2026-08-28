import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'

// The design system first, then this app's layer over it — order matters, the
// app sheet squares off the system's pill radii.
import './styles/organic.css'
import './styles/app.css'

import { App } from './App'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Most of these screens are forms; refetching under the user's hands
      // while they type would be worse than a slightly stale listing.
      refetchOnWindowFocus: false,
      staleTime: 15_000,
      // No automatic retry. A retry parks the query in react-query's "paused"
      // state whenever the tab is not focused — no data, no error, nothing a
      // screen can show, for as long as the tab stays in the background. This
      // console talks to one host on a LAN, so a failure is nearly always real:
      // report it at once and let the user press "Try again".
      retry: false,
      // Likewise, `networkMode: "online"` (the default) pauses instead of
      // fetching whenever the browser reports itself offline. The honest answer
      // to "can we reach the service" is the request's own outcome.
      networkMode: 'always',
    },
    mutations: { networkMode: 'always' },
  },
})

const root = document.getElementById('root')
if (!root) throw new Error('#root is missing from index.html')

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)
