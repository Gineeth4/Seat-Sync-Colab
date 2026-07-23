import { useEffect, useRef, useState, useCallback } from 'react'
import DetectPanel from './DetectPanel.jsx'

// The service is local and fixed. Override via Vite env if you ever move it.
const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
const WS_URL = (import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8000/ws/occupancy')

/**
 * Live occupancy dashboard.
 *
 * Data flow:
 *   GET /seats          -> static seat layout (normalised coords), fetched once
 *   WS  /ws/occupancy   -> per-frame occupancy pushed by the server producer
 *   GET /status         -> engine telemetry (fps, VRAM), polled every 2s
 *
 * The WebSocket auto-reconnects with a short backoff so the dashboard survives a
 * server restart without a manual refresh.
 */
export default function App() {
  const [seats, setSeats] = useState([])          // [{id, cx, cy, w, h}]
  const [occ, setOcc] = useState({})              // { seatId: {occupied, iou, occupant_conf} }
  const [meta, setMeta] = useState(null)          // { frame, occupied_count, seat_count, latency_ms }
  const [status, setStatus] = useState(null)      // { ema_fps, peak_vram_mb, frames_processed, device }
  const [conn, setConn] = useState('connecting')  // 'connecting' | 'live' | 'closed'
  const [mode, setMode] = useState('live')        // 'live' | 'detect'

  const wsRef = useRef(null)
  const retryRef = useRef(null)

  // --- one-time: seat layout ---
  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE}/seats`)
      .then((r) => r.json())
      .then((d) => { if (!cancelled) setSeats(d.seats || []) })
      .catch(() => {/* surfaced via the connection badge */})
    return () => { cancelled = true }
  }, [])

  // --- WebSocket with reconnect ---
  const connect = useCallback(() => {
    setConn('connecting')
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => setConn('live')
    ws.onmessage = (ev) => {
      const d = JSON.parse(ev.data)
      if (d.type !== 'occupancy') return
      const next = {}
      for (const s of d.seats) {
        next[s.seat_id] = { occupied: s.occupied, iou: s.iou, conf: s.occupant_conf }
      }
      setOcc(next)
      setMeta({
        frame: d.frame,
        occupied_count: d.occupied_count,
        seat_count: d.seat_count,
        latency_ms: d.latency_ms,
      })
    }
    ws.onclose = () => {
      setConn('closed')
      retryRef.current = setTimeout(connect, 1500) // backoff reconnect
    }
    ws.onerror = () => ws.close()
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  // --- status polling ---
  useEffect(() => {
    let alive = true
    const tick = () => {
      fetch(`${API_BASE}/status`)
        .then((r) => r.json())
        .then((d) => { if (alive) setStatus(d) })
        .catch(() => {})
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const occupiedCount = meta?.occupied_count ?? Object.values(occ).filter((s) => s.occupied).length
  const seatTotal = seats.length || meta?.seat_count || 0

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" /> Smart Seat Sync
          <span className="sub">seat intelligence</span>
        </div>
        <nav className="tabs">
          <button className={mode === 'live' ? 'tab active' : 'tab'} onClick={() => setMode('live')}>
            Live Occupancy
          </button>
          <button className={mode === 'detect' ? 'tab active' : 'tab'} onClick={() => setMode('detect')}>
            Detect Image
          </button>
        </nav>
        <div className="telemetry">
          <Stat label="occupied" value={`${occupiedCount} / ${seatTotal}`} accent />
          <Stat label="fps" value={status ? status.ema_fps.toFixed(1) : '—'} />
          <Stat label="vram" value={status ? `${status.peak_vram_mb.toFixed(0)} MB` : '—'} />
          <Stat label="frames" value={status ? status.frames_processed : '—'} />
          <Stat label="device" value={status ? `cuda:${status.device}` : '—'} />
          <ConnBadge state={conn} />
        </div>
      </header>

      {mode === 'detect' && (
        <main className="stage-wrap detect-wrap">
          <DetectPanel apiBase={API_BASE} />
        </main>
      )}

      {mode === 'live' && (<>
      <main className="stage-wrap">
        <div className="stage">
          {seats.length === 0 && (
            <div className="empty">Waiting for seat layout from {API_BASE}/seats…</div>
          )}
          {seats.map((s) => {
            const st = occ[s.id]
            const isOcc = !!st?.occupied
            return (
              <div
                key={s.id}
                className={`seat ${isOcc ? 'occ' : 'empty-seat'}`}
                style={{
                  left: `${(s.cx - s.w / 2) * 100}%`,
                  top: `${(s.cy - s.h / 2) * 100}%`,
                  width: `${s.w * 100}%`,
                  height: `${s.h * 100}%`,
                }}
              >
                <span className="seat-id">{s.id}</span>
                <span className="seat-state">{isOcc ? 'OCCUPIED' : 'empty'}</span>
                {st && st.iou > 0 && <span className="seat-iou">IoU {st.iou.toFixed(2)}</span>}
              </div>
            )
          })}
        </div>
      </main>

      <footer className="footbar">
        <span>frame: <b>{meta?.frame ?? '—'}</b></span>
        <span>inference: <b>{meta ? `${meta.latency_ms} ms` : '—'}</b></span>
        <span>source: <b>{status?.source ?? '—'}</b></span>
      </footer>
      </>)}
    </div>
  )
}

function Stat({ label, value, accent }) {
  return (
    <div className={`stat ${accent ? 'accent' : ''}`}>
      <div className="stat-val">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

function ConnBadge({ state }) {
  const text = { connecting: 'connecting', live: 'live', closed: 'reconnecting' }[state]
  return <div className={`conn ${state}`}><span className="conn-dot" />{text}</div>
}
