// Screen 2 — per-record trace. A left rail lists every processed record; the
// selected record's full six-touchpoint trace (with visible reasoning) renders
// on the right via the shared TraceView.
import React, { useEffect, useState } from 'react'
import TraceView from './TraceView.jsx'
import { Plate, StatusChip, TouchpointDots } from './bits.jsx'

export default function TraceScreen({ job }) {
  const records = job?.records || []
  const [activeId, setActiveId] = useState(records[0]?.record_id || null)

  // Default to the first record once results arrive; keep selection otherwise.
  useEffect(() => {
    if (!activeId && records.length) setActiveId(records[0].record_id)
  }, [records, activeId])

  const active = records.find((r) => r.record_id === activeId) || records[0]

  if (!records.length) {
    return (
      <div className="center-note">
        {job?.status === 'running' ? 'Processing the batch…' : 'No records to trace yet.'}
      </div>
    )
  }

  return (
    <div className="trace-layout">
      <aside className="rail-panel">
        <div className="rail-panel__head">
          <h2>Records</h2>
          <span className="rail-panel__count">
            {job.completed}/{job.total}
          </span>
        </div>
        <div className="rec-list">
          {records.map((r) => (
            <button
              key={r.record_id}
              className={`rec-item${r.record_id === active?.record_id ? ' rec-item--active' : ''}`}
              onClick={() => setActiveId(r.record_id)}
            >
              <div className="rec-item__top">
                <span className="rec-item__id">{r.record_id}</span>
                <TouchpointDots touchpoints={r.touchpoints} />
              </div>
              <div className="rec-item__raw">{r.raw_input}</div>
              <div className="rec-item__meta">
                <Plate id={r.match?.mmv_id} status={r.status} size="sm" />
                <StatusChip status={r.status} />
              </div>
            </button>
          ))}
        </div>
      </aside>

      {active && (
        <TraceView record={active} thresholds={job.thresholds} domPrefix="trace" key={active.record_id} />
      )}
    </div>
  )
}
