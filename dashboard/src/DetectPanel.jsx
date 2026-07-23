import { useCallback, useRef, useState } from 'react'

/**
 * Upload an image -> POST /detect -> show the annotated result + counts.
 * Drag & drop or click to pick a file. Uses the server's general (COCO) model,
 * so real photos of people are detected as `person`.
 */
export default function DetectPanel({ apiBase }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [fileName, setFileName] = useState(null)
  const [conf, setConf] = useState(0.25)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const submit = useCallback(async (file) => {
    if (!file) return
    setBusy(true); setError(null); setResult(null); setFileName(file.name)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${apiBase}/detect?conf=${conf}`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`server responded ${res.status}`)
      setResult(await res.json())
    } catch (e) {
      setError(e.message || 'detection failed')
    } finally {
      setBusy(false)
    }
  }, [apiBase, conf])

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) submit(file)
  }, [submit])

  return (
    <div className="detect">
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
          accept="image/*"
          hidden
          onChange={(e) => submit(e.target.files?.[0])}
        />
        <div className="dz-icon">⬆</div>
        <div className="dz-title">{busy ? 'Detecting…' : 'Drop an image here, or click to choose'}</div>
        <div className="dz-sub">{fileName ? `last: ${fileName}` : 'a photo of people / students works best'}</div>
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

      {result && (
        <div className="detect-result">
          <div className="result-summary">
            <span className="pill person">{result.person_count} person</span>
            <span className="pill">{result.n_detections} total</span>
            {Object.entries(result.counts)
              .filter(([k]) => k !== 'person')
              .map(([k, v]) => <span className="pill muted" key={k}>{v} {k}</span>)}
          </div>
          <img className="annotated" src={result.annotated_image} alt="detections" />
        </div>
      )}
    </div>
  )
}
