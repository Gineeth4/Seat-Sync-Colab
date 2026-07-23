import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The FastAPI service (with its CORS allow-list for :5173) runs separately on
// :8000, so no proxy is needed — the app talks to it directly. Keeping the dev
// server on the fixed port the backend already whitelists avoids a CORS surprise.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
})
