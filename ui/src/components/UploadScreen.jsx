// Screen 1 — upload the master catalogue + input batch, choose a mode, and
// start a matching run.
import React, { useRef, useState } from 'react'
import { getSample } from '../api.js'

export default function UploadScreen({ health, busy, error, onStart }) {
  const [master, setMaster] = useState(null)
  const [inputs, setInputs] = useState(null)
  const [mode, setMode] = useState('mock')
  const [sampleLoading, setSampleLoading] = useState(false)

  const liveAvailable = !!health?.gemini_key_present

  const loadSample = async () => {
    setSampleLoading(true)
    try {
      const { csv, filename } = await getSample()
      const file = new File([csv], filename, { type: 'text/csv' })
      setInputs(withRowCount(file, csv))
    } catch {
      /* ignore; user can still upload manually */
    } finally {
      setSampleLoading(false)
    }
  }

  const canStart = !!inputs && !busy

  return (
    <div className="upload">
      <div className="hero">
        <div className="hero__eyebrow">Make · Model · Variant reconciliation</div>
        <h1>
          Match messy vehicle strings to the catalogue — and <em>audit every step</em> of
          the reasoning.
        </h1>
        <p>
          Upload a batch of free-text inputs. Each one runs through six touchpoints —
          normalization, query reformulation, ReAct reasoning, an adversarial verifier,
          and explanation — with the full decision trace kept on the record.
        </p>
      </div>

      <div className="dropzones">
        <Dropzone
          label="Master catalogue"
          tag="optional"
          title="MMV master CSV"
          hint={`The canonical vehicle list to match against. Leave empty to use the bundled catalogue${
            health?.default_master_rows ? ` (${health.default_master_rows} rows)` : ''
          }.`}
          file={master}
          onFile={setInputsFactory(setMaster)}
          onClear={() => setMaster(null)}
        />
        <Dropzone
          label="Input batch"
          tag="required"
          required
          title="Vehicle strings CSV"
          hint="One column of raw vehicle strings (a raw_input column, or the first column). An optional input_id column is used as the record id."
          file={inputs}
          onFile={setInputsFactory(setInputs)}
          onClear={() => setInputs(null)}
          extra={
            <button className="link-btn" onClick={loadSample} disabled={sampleLoading} type="button">
              {sampleLoading ? 'loading…' : 'load sample batch'}
            </button>
          }
        />
      </div>

      <div className="options">
        <span className="opt-title">Engine</span>
        <div className="toggle" role="tablist">
          <button
            className={mode === 'mock' ? 'on' : ''}
            onClick={() => setMode('mock')}
            type="button"
          >
            <span className="swatch swatch--mock" /> Simulated
          </button>
          <button
            className={mode === 'live' ? 'on' : ''}
            onClick={() => liveAvailable && setMode('live')}
            disabled={!liveAvailable}
            type="button"
            title={liveAvailable ? '' : 'Set GEMINI_API_KEY in .env to enable live mode'}
          >
            <span className="swatch swatch--live" /> Live LLM
          </button>
        </div>
        <span className="opt-note">
          {mode === 'mock'
            ? 'Deterministic, no-LLM run using the real retrieval + validation tools. Reasoning is synthesized and clearly flagged — no Gemini quota spent.'
            : 'Runs the full Gemini pipeline in graph.py. Uses your API key and can be rate-limited on free tiers.'}
          {!liveAvailable && ' Live mode needs GEMINI_API_KEY.'}
        </span>
      </div>

      {error && <div className="upload__err">{error}</div>}

      <div className="upload__actions">
        <button
          className="btn btn--primary"
          disabled={!canStart}
          onClick={() => onStart({ inputsFile: inputs, masterFile: master, mode })}
        >
          {busy ? 'Starting…' : 'Start matching'} <span aria-hidden>→</span>
        </button>
        {!inputs && <span style={{ fontSize: 13, color: 'var(--muted)' }}>Add an input batch to begin.</span>}
      </div>
    </div>
  )
}

function Dropzone({ label, tag, required, title, hint, file, onFile, onClear, extra }) {
  const [over, setOver] = useState(false)
  const inputRef = useRef(null)
  const cls = `dz${over ? ' dz--over' : ''}${file ? ' dz--filled' : ''}`
  return (
    <div
      className={cls}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const f = e.dataTransfer.files?.[0]
        if (f) onFile(f)
      }}
    >
      <div className="dz__label">
        {label}
        <span className={`dz__tag${required ? ' dz__tag--req' : ''}`}>{tag}</span>
      </div>
      <h3>{title}</h3>
      <p className="dz__hint">{hint}</p>
      {file ? (
        <div className="dz__file">
          <span aria-hidden>▤</span>
          <span className="name">{file.name}</span>
          {file._rows != null && <span className="rows">{file._rows} rows</span>}
          <button className="x" onClick={onClear} type="button" aria-label="Remove file">
            ✕
          </button>
        </div>
      ) : (
        <div className="dz__browse">
          <button className="btn btn--ghost btn--sm" type="button" onClick={() => inputRef.current?.click()}>
            Browse CSV
          </button>
        </div>
      )}
      {extra && !file && <div style={{ marginTop: 10 }}>{extra}</div>}
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
        }}
      />
    </div>
  )
}

// Wrap the raw File with a best-effort row count for display.
function setInputsFactory(setter) {
  return (file) => {
    const reader = new FileReader()
    reader.onload = () => setter(withRowCount(file, String(reader.result)))
    reader.onerror = () => setter(file)
    reader.readAsText(file)
  }
}

function withRowCount(file, text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0)
  file._rows = Math.max(0, lines.length - 1) // minus header
  return file
}
