// Screen 3 — results table. One row per record (vehicle, match, confidence,
// status, touchpoints fired, verifier outcome). A row expands to the full trace
// plus a human override control.
import React, { useState } from 'react'
import TraceView from './TraceView.jsx'
import { Plate, StatusChip, ConfidenceMeter, TouchpointDots } from './bits.jsx'
import { overrideRecord } from '../api.js'

export default function ResultsScreen({ job, onRecordPatched }) {
  const records = job?.records || []
  const [openId, setOpenId] = useState(null)

  if (!records.length) {
    return (
      <div className="center-note">
        {job?.status === 'running' ? 'Processing the batch…' : 'No results yet.'}
      </div>
    )
  }

  const tally = { approved: 0, review: 0, rejected: 0 }
  records.forEach((r) => {
    const s = r.override?.decision || r.status
    tally[s] = (tally[s] || 0) + 1
  })

  return (
    <div className="results">
      <div className="results__head">
        <div>
          <h2>Results</h2>
          <p>
            {records.length} record{records.length === 1 ? '' : 's'} · {job.mode} engine ·{' '}
            click a row for its full trace and to override the decision
          </p>
        </div>
        <div className="tallies">
          <Tally n={tally.approved} label="approved" color="var(--ok)" />
          <Tally n={tally.review} label="review" color="var(--warn)" />
          <Tally n={tally.rejected} label="rejected" color="var(--bad)" />
        </div>
      </div>

      <div className="table-wrap">
        <table className="rtable">
          <thead>
            <tr>
              <th className="chev-cell" />
              <th>Vehicle input</th>
              <th>Match</th>
              <th>Confidence</th>
              <th>Status</th>
              <th className="hide-sm">Touchpoints</th>
              <th>Verifier</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r) => {
              const open = openId === r.record_id
              return (
                <React.Fragment key={r.record_id}>
                  <tr
                    className={`row${open ? ' row--open' : ''}`}
                    onClick={() => setOpenId(open ? null : r.record_id)}
                    role="button"
                    tabIndex={0}
                    aria-expanded={open}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        setOpenId(open ? null : r.record_id)
                      }
                    }}
                  >
                    <td className="chev-cell">
                      <span className={open ? 'open' : ''}>▶</span>
                    </td>
                    <td className="cell-input">
                      <div className="raw">{r.raw_input}</div>
                      <div className="rid">
                        {r.record_id}
                        {r.override && <span className="overridden-flag">overridden</span>}
                      </div>
                    </td>
                    <td>
                      <Plate id={r.match?.mmv_id} status={r.status} size="sm" />
                    </td>
                    <td className="cell-conf">
                      <ConfidenceMeter
                        confidence={r.confidence}
                        tier={r.confidence_tier}
                        thresholds={job.thresholds}
                      />
                    </td>
                    <td>
                      <StatusChip status={r.override?.decision || r.status} />
                    </td>
                    <td className="hide-sm">
                      <TouchpointDots touchpoints={r.touchpoints} />
                    </td>
                    <td className="cell-vf">
                      <VerifierCell verifier={r.verifier} />
                    </td>
                  </tr>
                  {open && (
                    <tr>
                      <td className="expand-td" colSpan={7}>
                        <div className="expand-inner">
                          <TraceView record={r} thresholds={job.thresholds} domPrefix="res" />
                          <OverridePanel job={job} record={r} onRecordPatched={onRecordPatched} />
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Tally({ n, label, color }) {
  return (
    <div className="tally">
      <i style={{ background: color }} />
      <span className="tally__n">{n}</span>
      <span className="tally__l">{label}</span>
    </div>
  )
}

function VerifierCell({ verifier }) {
  if (!verifier?.fired) return <span className="vf--na">— not run —</span>
  return verifier.passed ? (
    <span className="vf--pass">✓ passed</span>
  ) : (
    <span className="vf--fail">✕ concern</span>
  )
}

function OverridePanel({ job, record, onRecordPatched }) {
  const [decision, setDecision] = useState(record.override?.decision || record.status)
  const [note, setNote] = useState(record.override?.note || '')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(!!record.override)
  const [err, setErr] = useState(null)

  const save = async () => {
    setSaving(true)
    setErr(null)
    try {
      const updated = await overrideRecord(job.job_id, record.record_id, decision, note)
      onRecordPatched(updated)
      setSaved(true)
    } catch (e) {
      setErr(e.message)
    } finally {
      setSaving(false)
    }
  }

  const dirty = decision !== (record.override?.decision || record.status) || note !== (record.override?.note || '')

  return (
    <div className="override">
      <h4>Human override</h4>
      <p className="sub">
        Reviewers have the final say. Set the decision this record should carry and leave a note for the
        audit trail.
      </p>
      <div className="override__row">
        <div className="field">
          <label>Decision</label>
          <div className="seg">
            {['approved', 'review', 'rejected'].map((d) => (
              <button
                key={d}
                className={decision === d ? `on--${d}` : ''}
                onClick={() => {
                  setDecision(d)
                  setSaved(false)
                }}
                type="button"
              >
                {d}
              </button>
            ))}
          </div>
        </div>
        <div className="field" style={{ flex: 1 }}>
          <label>Note</label>
          <input
            type="text"
            value={note}
            placeholder="e.g. confirmed VXI manual against the source doc"
            onChange={(e) => {
              setNote(e.target.value)
              setSaved(false)
            }}
          />
        </div>
        <button className="btn btn--primary btn--sm" onClick={save} disabled={saving || (!dirty && saved)}>
          {saving ? 'Saving…' : saved && !dirty ? 'Saved' : 'Save override'}
        </button>
      </div>
      {err && <div className="upload__err" style={{ marginTop: 12 }}>{err}</div>}
      {saved && !dirty && record.override && (
        <div className="override__saved">
          ✓ Overridden to <b>&nbsp;{record.override.decision}</b> · logged {formatTime(record.override.at)}
        </div>
      )}
    </div>
  )
}

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString()
  } catch {
    return iso
  }
}
