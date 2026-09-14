import { useCallback, useMemo, useRef, useState } from 'react'
import SeatGrid from './SeatGrid.jsx'

const VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
// Passed explicitly to both /detect/video and /occupancy/video so they sample
// the same frame indices — lets one activeFrame index drive both timelines.
const VIDEO_MAX_FRAMES = 24

function isVideoFile(file) {
  if (!file) return false
  if (file.type?.startsWith('video/')) return true
  const name = file.name?.toLowerCase() || ''
  return VIDEO_EXTS.some((ext) => name.endsWith(ext))
}

function seatsToOccMap(seatStates) {
  const map = {}
  for (const s of seatStates || []) {
    map[s.seat_id] = { occupied: s.occupied, iou: s.iou, conf: s.occupant_conf }
  }
  return map
}

/**
 * Upload an image or video -> general (COCO) detection + per-seat occupancy
 * (using the trained seat model + calibrated layout), shown side by side:
 * annotated boxes/counts from /detect, and the same seat grid the Live tab
 * uses from /occupancy. Drag & drop or click to pick a file.
 */
export default function DetectPanel({ apiBase, seats = [] }) {
  const [kind, setKind] = useState('image') // 'image' | 'video'
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [occError, setOccError] = useState(null)
  const [result, setResult] = useState(null)
  const [occResult, setOccResult] = useState(null)
  const [videoResult, setVideoResult] = useState(null)
  const [videoOccResult, setVideoOccResult] = useState(null)
  const [activeFrame, setActiveFrame] = useState(0)
  const [fileName, setFileName] = useState(null)
  const [conf, setConf] = useState(0.25)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const submitImage = useCallback(async (file) => {
    setBusy(true); setError(null); setOccError(null)
    setResult(null); setOccResult(null); setVideoResult(null); setVideoOccResult(null)
    setFileName(file.name)
    try {
      const [detectRes, occRes] = await Promise.all([
        fetch(`${apiBase}/detect?conf=${conf}`, { method: 'POST', body: (() => {
          const f = new FormData(); f.append('file', file); return f
        })() }),
        fetch(`${apiBase}/occupancy?conf=${conf}`, { method: 'POST', body: (() => {
          const f = new FormData(); f.append('file', file); return f
        })() }),
      ])
      if (!detectRes.ok) throw new Error(`server responded ${detectRes.status}`)
      setResult(await detectRes.json())

      if (occRes.ok) {
        setOccResult(await occRes.json())
      } else {
        const body = await occRes.json().catch(() => null)
        setOccError(body?.detail || `seat occupancy unavailable (${occRes.status})`)
      }
    } catch (e) {
      setError(e.message || 'detection failed')
    } finally {
      setBusy(false)
    }
  }, [apiBase, conf])

  const submitVideo = useCallback(async (file) => {
    setBusy(true); setError(null); setOccError(null)
    setResult(null); setOccResult(null); setVideoResult(null); setVideoOccResult(null)
    setFileName(file.name)
    setActiveFrame(0)
    try {
      const [detectRes, occRes] = await Promise.all([
        fetch(`${apiBase}/detect/video?conf=${conf}&max_frames=${VIDEO_MAX_FRAMES}`, {
          method: 'POST', body: (() => { const f = new FormData(); f.append('file', file); return f })(),
        }),
        fetch(`${apiBase}/occupancy/video?conf=${conf}&max_frames=${VIDEO_MAX_FRAMES}`, {
          method: 'POST', body: (() => { const f = new FormData(); f.append('file', file); return f })(),
        }),
      ])
      if (!detectRes.ok) {
        const body = await detectRes.json().catch(() => null)
        throw new Error(body?.detail || `server responded ${detectRes.status}`)
      }
      setVideoResult(await detectRes.json())

      if (occRes.ok) {
        setVideoOccResult(await occRes.json())
      } else {
        const body = await occRes.json().catch(() => null)
        setOccError(body?.detail || `seat occupancy unavailable (${occRes.status})`)
      }
    } catch (e) {
      setError(e.message || 'video detection failed')
    } finally {
      setBusy(false)
    }
  }, [apiBase, conf])

  const submit = useCallback((file) => {
    if (!file) return
    if (isVideoFile(file)) {
      setKind('video')
      submitVideo(file)
    } else {
      setKind('image')
      submitImage(file)
    }
  }, [submitImage, submitVideo])

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) submit(file)
  }, [submit])

  const frame = videoResult?.timeline?.[activeFrame]
  const occFrame = videoOccResult?.timeline?.[activeFrame]

  const imageOccMap = useMemo(() => seatsToOccMap(occResult?.seats), [occResult])
  const videoOccMap = useMemo(() => seatsToOccMap(occFrame?.seats), [occFrame])

  return (
    <div className="detect">
      <div className="mode-row">
        <button
          className={kind === 'image' ? 'tab active' : 'tab'}
          onClick={() => { setKind('image'); setError(null) }}
        >
          Image
        </button>
        <button
          className={kind === 'video' ? 'tab active' : 'tab'}
          onClick={() => { setKind('video'); setError(null) }}
        >
          Video
        </button>
      </div>

      <div
        className={`dropzone ${dragging ? 'drag' : ''} ${busy ? 'busy' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept={kind === 'video' ? 'video/*' : 'image/*'}
          hidden
          onChange={(e) => submit(e.target.files?.[0])}
        />
        <div className="dz-icon">⬆</div>
        <div className="dz-title">
          {busy
            ? (kind === 'video' ? 'Scanning frames…' : 'Detecting…')
            : `Drop a ${kind === 'video' ? 'video' : 'image'} here, or click to choose`}
        </div>
        <div className="dz-sub">
          {fileName
            ? `last: ${fileName}`
            : kind === 'video'
              ? 'mp4 / mov / avi / mkv / webm — frames are sampled evenly across the clip'
              : 'a photo with people, chairs, or study tables works best'}
        </div>
      </div>

      <div className="conf-row">
        <label>confidence: <b>{conf.toFixed(2)}</b></label>
        <input
          type="range" min="0.05" max="0.8" step="0.05"
          value={conf}
          onChange={(e) => setConf(parseFloat(e.target.value))}
        />
        <span className="conf-hint">lower = more (but noisier) detections</span>
      </div>

      {error && <div className="detect-error">⚠ {error} — is the server running on {apiBase}?</div>}

      {kind === 'image' && result && (
        <div className="detect-result">
          <div className="result-summary">
            {result.person_count !== undefined && <span className="pill person">{result.person_count} person</span>}
            <span className="pill">{result.n_detections} total</span>
            {Object.entries(result.counts)
              .filter(([k]) => k !== 'person')
              .map(([k, v]) => (
                <span
                  className={`pill ${k === 'chair' ? 'chair' : k === 'table' ? 'table' : 'muted'}`}
                  key={k}
                >
                  {v} {k}
                </span>
              ))}
          </div>
          <img className="annotated" src={result.annotated_image} alt="detections" />

          <SeatsSection
            seats={seats}
            occMap={imageOccMap}
            occResult={occResult}
            occError={occError}
          />
        </div>
      )}

      {kind === 'video' && videoResult && (
        <div className="detect-result">
          <div className="result-summary">
            <span className="pill person">peak {videoResult.peak_person_count} person</span>
            <span className="pill">avg {videoResult.avg_person_count} person/frame</span>
            <span className="pill muted">{videoResult.sampled_frames} of {videoResult.total_frames} frames sampled</span>
            {videoResult.duration_sec != null && (
              <span className="pill muted">{videoResult.duration_sec.toFixed(1)}s clip</span>
            )}
          </div>

          {frame && (
            <>
              <img className="annotated" src={frame.annotated_image} alt={`frame ${frame.frame_index}`} />
              <div className="result-summary">
                <span className="pill person">{frame.person_count} person</span>
                <span className="pill">{frame.n_detections} total</span>
                {Object.entries(frame.counts)
                  .filter(([k]) => k !== 'person')
                  .map(([k, v]) => (
                    <span
                      className={`pill ${k === 'chair' ? 'chair' : k === 'table' ? 'table' : 'muted'}`}
                      key={k}
                    >
                      {v} {k}
                    </span>
                  ))}
                <span className="pill muted">
                  {frame.timestamp_sec != null ? `t=${frame.timestamp_sec.toFixed(1)}s` : `frame ${frame.frame_index}`}
                </span>
              </div>
            </>
          )}

          {videoResult.timeline.length > 1 && (
            <div className="timeline">
              {videoResult.timeline.map((f, i) => (
                <button
                  key={f.frame_index}
                  className={`timeline-thumb ${i === activeFrame ? 'active' : ''}`}
                  onClick={() => setActiveFrame(i)}
                  title={f.timestamp_sec != null ? `t=${f.timestamp_sec.toFixed(1)}s` : `frame ${f.frame_index}`}
                >
                  <img src={f.annotated_image} alt={`frame ${f.frame_index}`} />
                  {f.person_count > 0 && <span className="timeline-badge">{f.person_count}</span>}
                </button>
              ))}
            </div>
          )}

          <SeatsSection
            seats={seats}
            occMap={videoOccMap}
            occResult={occFrame}
            occError={occError}
          />
        </div>
      )}
    </div>
  )
}

function SeatsSection({ seats, occMap, occResult, occError }) {
  return (
    <div className="seats-section">
      <div className="seats-section-head">
        <h3>Seats</h3>
        {occResult && (
          <span className="pill occ">{occResult.occupied_count} / {occResult.seat_count} occupied</span>
        )}
      </div>
      {occError && <div className="detect-error">⚠ {occError}</div>}
      {!occError && seats.length === 0 && (
        <div className="dz-sub">No seat layout configured (GET /seats returned none).</div>
      )}
      {!occError && seats.length > 0 && !occResult && (
        <div className="dz-sub">Waiting for occupancy result…</div>
      )}
      {!occError && seats.length > 0 && occResult && (
        <div className="seat-grid-card">
          <SeatGrid seats={seats} occ={occMap} />
        </div>
      )}
    </div>
  )
}
