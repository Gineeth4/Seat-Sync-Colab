/**
 * Renders the configured seat layout as a grid of boxes, coloured by
 * occupancy. Shared between the live view and the Detect panel (image/video
 * uploads), so both read from the same seat calibration.
 *
 * seats: [{id, cx, cy, w, h}]  (normalised 0-1, from GET /seats)
 * occ:   { [seatId]: { occupied, iou, conf } }
 */
export default function SeatGrid({ seats, occ, emptyMessage }) {
  return (
    <div className="stage">
      {seats.length === 0 && (
        <div className="empty">{emptyMessage || 'No seat layout available.'}</div>
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
  )
}
