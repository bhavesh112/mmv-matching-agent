// Small shared presentational pieces: the number-plate MMV token (signature),
// status chips, the confidence meter, and the touchpoint dot strip.
import React from 'react'

const STATUS_TO_PLATE = { approved: 'approved', review: 'review', rejected: 'rejected' }

// The signature element: an MMV id rendered as an Indian-style number plate.
// White plate = approved, amber (commercial) plate = review, hatched = rejected.
export function Plate({ id, status, size }) {
  if (!id) {
    return <span className="plate--none">— no match —</span>
  }
  const variant = STATUS_TO_PLATE[status] || 'approved'
  return (
    <span className={`plate plate--${variant}${size === 'sm' ? ' plate--sm' : ''}`}>
      <span className="plate__tab">IND</span>
      <span className="plate__id">{id}</span>
    </span>
  )
}

export function StatusChip({ status }) {
  const label = status || 'review'
  return (
    <span className={`chip chip--${label}`}>
      <span className="dot" />
      {label}
    </span>
  )
}

const TIER_LABEL = { auto: 'auto-approve tier', review: 'review band', reject: 'reject band' }

// Segmented confidence bar with the three routing tiers drawn to scale and a
// marker at the record's confidence. Thresholds come from the run.
export function ConfidenceMeter({ confidence, tier, thresholds }) {
  const floor = thresholds?.review_floor ?? 0.8
  const auto = thresholds?.auto_approve ?? 0.95
  const pct = Math.max(0, Math.min(1, confidence)) * 100
  return (
    <div className="conf">
      <div className="conf__head">
        <span className="conf__val">{confidence.toFixed(2)}</span>
        <span className="conf__tier">{TIER_LABEL[tier] || tier}</span>
      </div>
      <div className="conf__track" title={`confidence ${confidence.toFixed(2)}`}>
        <span className="conf__seg conf__seg--reject" style={{ width: `${floor * 100}%` }} />
        <span className="conf__seg conf__seg--review" style={{ width: `${(auto - floor) * 100}%` }} />
        <span className="conf__seg conf__seg--auto" style={{ width: `${(1 - auto) * 100}%` }} />
        <span className="conf__marker" style={{ left: `${pct}%` }} />
      </div>
    </div>
  )
}

// Compact 6-dot strip summarizing which touchpoints fired (used in lists/table).
export function TouchpointDots({ touchpoints }) {
  return (
    <span className="mini-dots" title="Touchpoints fired (1–6)">
      {touchpoints.map((t) => {
        const cls =
          t.status === 'fired' ? 'on' : t.status === 'error' ? 'err' : t.status === 'offline' ? 'off' : ''
        return <i key={t.id} className={cls} title={`${t.id}. ${t.name} — ${t.status}`} />
      })}
    </span>
  )
}
