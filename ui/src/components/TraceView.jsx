// The per-record trace: a pipeline rail of the six touchpoints, followed by an
// expandable card per touchpoint showing its individual output — including the
// full ReAct reasoning tape (thought -> tool call -> observation). Reused by
// both the Trace screen and the expanded rows on the Results screen.
import React, { useState } from 'react'
import { Plate, StatusChip, ConfidenceMeter } from './bits.jsx'

const STATE_LABEL = {
  fired: 'fired',
  skipped: 'skipped',
  offline: 'offline',
  error: 'error',
}

export default function TraceView({ record, thresholds, domPrefix = 'tp' }) {
  // Reasoning (touchpoint 3) is the point of the tool, so open it by default.
  const [open, setOpen] = useState(() => new Set([3]))
  const toggle = (id) =>
    setOpen((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })

  const jump = (tp) => {
    if (tp.status !== 'fired') return
    setOpen((prev) => new Set(prev).add(tp.id))
    requestAnimationFrame(() => {
      const el = document.getElementById(`${domPrefix}-${record.record_id}-${tp.id}`)
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    })
  }

  return (
    <div className="trace-main">
      <DecisionCard record={record} thresholds={thresholds} />
      <PipelineRail touchpoints={record.touchpoints} onJump={jump} />
      <div className="tp-cards">
        {record.touchpoints.map((tp) => (
          <TouchpointCard
            key={tp.id}
            tp={tp}
            record={record}
            isOpen={open.has(tp.id)}
            onToggle={() => toggle(tp.id)}
            domId={`${domPrefix}-${record.record_id}-${tp.id}`}
          />
        ))}
      </div>
    </div>
  )
}

function DecisionCard({ record, thresholds }) {
  const status = record.status
  return (
    <div className="decision-card">
      <div className={`decision-card__bar bar--${status}`} />
      <div className="decision-card__body">
        <div className="dc-head">
          <div className="dc-head__main">
            <span className="dc-input-label">Input string</span>
            <div className="dc-raw">{record.raw_input}</div>
            <div className="dc-row">
              <Plate id={record.match?.mmv_id} status={status} />
              <span className="dc-arrow">→</span>
              <StatusChip status={status} />
              {record.match && (
                <span style={{ fontSize: 13, color: 'var(--muted)' }}>
                  {describeMatch(record.match)}
                </span>
              )}
              {record.override && (
                <span className="overridden-flag">overridden → {record.override.decision}</span>
              )}
            </div>
          </div>
          <div className="dc-side">
            <ConfidenceMeter
              confidence={record.confidence}
              tier={record.confidence_tier}
              thresholds={thresholds}
            />
          </div>
        </div>
        {record.explanation && (
          <div className="dc-explain">
            <span className="lead">Explanation · touchpoint 5</span>
            {record.explanation}
          </div>
        )}
      </div>
    </div>
  )
}

function PipelineRail({ touchpoints, onJump }) {
  return (
    <div className="rail" role="list" aria-label="Pipeline touchpoints">
      {touchpoints.map((tp) => (
        <div
          key={tp.id}
          role="listitem"
          className={`station station--${tp.status}${tp.status === 'fired' ? ' station--clickable' : ''}`}
          onClick={() => onJump(tp)}
        >
          <div className="station__top">
            <span className="station__num">{tp.id}</span>
            <span className="station__name">{tp.name}</span>
            {tp.status !== 'fired' && (
              <span className="station__state">{STATE_LABEL[tp.status]}</span>
            )}
          </div>
          <span className="station__sum">{tp.summary}</span>
        </div>
      ))}
    </div>
  )
}

function TouchpointCard({ tp, record, isOpen, onToggle, domId }) {
  const canOpen = tp.status === 'fired'
  return (
    <div className={`tp-card tp-card--${tp.status}`} id={domId}>
      <button
        className="tp-card__head"
        onClick={canOpen ? onToggle : undefined}
        aria-expanded={canOpen ? isOpen : undefined}
      >
        <span className="tp-card__badge">{tp.id}</span>
        <span style={{ minWidth: 0 }}>
          <span className="tp-card__title">{tp.name}</span>
          <div className="tp-card__sum">{tp.summary}</div>
        </span>
        <span className="tp-card__state">{STATE_LABEL[tp.status]}</span>
        {canOpen && <span className={`tp-card__chev${isOpen ? ' open' : ''}`}>▶</span>}
      </button>
      {canOpen && isOpen && (
        <div className="tp-card__body">
          <TouchpointOutput tp={tp} record={record} />
          {tp.calls && tp.calls.length > 0 && (
            <div className="tp-card__calls">
              {tp.calls.map((c, i) => (
                <span key={i}>
                  {c.model} · {c.latency_s}s
                  {c.total_tokens ? ` · ${c.total_tokens} tok` : ''}
                  {c.ok === false ? ' · failed' : ''}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// --- Per-touchpoint output renderers --------------------------------------

function TouchpointOutput({ tp, record }) {
  switch (tp.key) {
    case 'normalization':
      return <NormalizationOut output={tp.output} />
    case 'query_reformulation':
      return <QueriesOut output={tp.output} />
    case 'reasoning':
      return <ReasoningOut record={record} />
    case 'verifier':
      return <VerifierOut output={tp.output} />
    case 'explanation':
      return <p className="obs--text" style={{ marginTop: 10 }}>{tp.output?.explanation}</p>
    default:
      return null
  }
}

const NORM_FIELDS = [
  ['make', 'Make'], ['model', 'Model'], ['variant', 'Variant'], ['fuel_type', 'Fuel'],
  ['transmission', 'Transmission'], ['cc', 'CC'], ['seating_capacity', 'Seating'], ['year', 'Year'],
]

function NormalizationOut({ output }) {
  const rec = output?.normalized_record || {}
  const syn = rec.applied_synonyms || []
  return (
    <>
      <div className="attr-grid">
        {NORM_FIELDS.map(([k, label]) => (
          <div className="attr-cell" key={k}>
            <div className="attr-cell__k">{label}</div>
            <div className={`attr-cell__v${rec[k] == null || rec[k] === '' ? ' empty' : ''}`}>
              {rec[k] == null || rec[k] === '' ? 'not stated' : String(rec[k])}
            </div>
          </div>
        ))}
      </div>
      {syn.length > 0 && (
        <div className="syn-list">
          {syn.map((s, i) => (
            <span className="syn" key={i}>
              <s>{s.from}</s> → <b>{s.to}</b> <span className="cat">{s.category}</span>
            </span>
          ))}
        </div>
      )}
    </>
  )
}

function QueriesOut({ output }) {
  const queries = output?.queries || []
  return (
    <div className="query-list">
      {queries.map((q, i) => (
        <div className="query" key={i}>
          <span className="qn">Q{i + 1}</span>
          {q}
        </div>
      ))}
    </div>
  )
}

function ReasoningOut({ record }) {
  const trace = record.reasoning?.trace || []
  return (
    <>
      <div style={{ marginTop: 10, fontSize: 13.5, color: 'var(--muted)', lineHeight: 1.5 }}>
        {record.reasoning?.step_count} step(s) · hard-rule validation:{' '}
        <span className={`val-pill val-pill--${record.validation_status === 'pass' ? 'pass' : 'fail'}`}>
          {record.validation_status || 'n/a'}
        </span>
      </div>
      <ReasoningTape trace={trace} />
    </>
  )
}

function ReasoningTape({ trace }) {
  return (
    <div className="tape">
      {trace.map((step, i) => {
        const isFinish = step.action === 'finish'
        return (
          <div className={`tape-step${isFinish ? ' tape-step--finish' : ''}`} key={i}>
            <div className="tape-step__node">{isFinish ? '✓' : step.step}</div>
            {step.thought && <div className="tape-step__thought">{step.thought}</div>}
            {step.action && (
              <div className="tape-step__action">
                <ToolCall action={step.action} actionInput={step.action_input} />
              </div>
            )}
            {step.observation != null && step.observation !== '' && (
              <div className="tape-step__obs">
                <Observation obs={step.observation} />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function ToolCall({ action, actionInput }) {
  const args = actionInput && typeof actionInput === 'object'
    ? Object.entries(actionInput)
        .map(([k, v]) => `${k}: ${v == null ? '∅' : v}`)
        .join(', ')
    : ''
  return (
    <span className={`tool-call${action === 'finish' ? ' tool-call--finish' : ''}`}>
      <span className="fn">{action}</span>
      {args && <span className="args">{args}</span>}
    </span>
  )
}

// Renders a tool observation: attribute_compare diff grid, validate result,
// a plain string, or (fallback) pretty JSON.
function Observation({ obs }) {
  if (typeof obs === 'string') {
    return <div className="obs obs--text">{obs}</div>
  }
  if (obs && Array.isArray(obs.attributes)) {
    return <AttributeDiff obs={obs} />
  }
  if (obs && obs.validation_status) {
    return <ValidateObs obs={obs} />
  }
  return (
    <div className="obs">
      <pre className="obs__raw">{JSON.stringify(obs, null, 2)}</pre>
    </div>
  )
}

function AttributeDiff({ obs }) {
  return (
    <div className="obs">
      <div className="obs__title">attribute_compare · {obs.mmv_id}</div>
      <div className="diff">
        <div className="diff-row diff-head">
          <span>Attribute</span>
          <span>Input</span>
          <span>Candidate</span>
          <span style={{ textAlign: 'right' }}>Match</span>
        </div>
        {obs.attributes.map((a) => (
          <div className={`diff-row${a.status === 'mismatch' ? ' diff-row--mismatch' : ''}`} key={a.attribute}>
            <span className="diff-row__attr">{a.attribute.replace('_', ' ')}</span>
            <span className={`diff-row__val${a.input == null ? ' empty' : ''}`}>
              {fmt(a.input)}
            </span>
            <span className={`diff-row__val${a.candidate == null ? ' empty' : ''}`}>
              {fmt(a.candidate)}
            </span>
            <span className={`diff-row__st st--${a.status}`}>{shortStatus(a.status)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function ValidateObs({ obs }) {
  const pass = obs.validation_status === 'pass'
  return (
    <div className="obs">
      <div className="obs__title">
        validate
        <span className={`val-pill val-pill--${pass ? 'pass' : 'fail'}`}>{obs.validation_status}</span>
      </div>
      {obs.violations && obs.violations.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {obs.violations.map((v, i) => (
            <div className="viol" key={i}>
              ✗ {v.rule}: {v.detail}
            </div>
          ))}
        </div>
      ) : (
        <div style={{ fontSize: 12.5, color: 'var(--ok)' }}>No hard-rule violations.</div>
      )}
      {obs.checked && obs.checked.length > 0 && (
        <div className="checked">
          rules checked: <b>{obs.checked.join(', ')}</b>
        </div>
      )}
    </div>
  )
}

function VerifierOut({ output }) {
  if (!output) return null
  const pass = output.passed
  return (
    <div className={`verifier verifier--${pass ? 'pass' : 'fail'}`}>
      <div className="verifier__verdict">
        <span className={`val-pill val-pill--${pass ? 'pass' : 'fail'}`}>verdict: {output.verdict}</span>
        <span style={{ fontSize: 12.5, color: 'var(--muted)' }}>
          {pass ? 'match survived the adversarial challenge' : 'downgraded to human review'}
        </span>
      </div>
      <div className="verifier__concern">
        <span className="verifier__lead">strongest argument against the match</span>
        <div style={{ marginTop: 5 }}>{output.concern}</div>
      </div>
    </div>
  )
}

// --- helpers ---------------------------------------------------------------

function describeMatch(m) {
  return [m.make, m.model, m.variant, m.fuel_type, m.transmission].filter(Boolean).join(' ')
}

function fmt(v) {
  if (v == null) return 'not stated'
  if (Array.isArray(v)) return v.map((x) => (x == null ? '·' : x)).join('–')
  return String(v)
}

function shortStatus(s) {
  return { match: 'yes', mismatch: 'no', input_missing: '—', candidate_missing: '?' }[s] || s
}
