import React, { useCallback, useEffect, useRef, useState } from 'react'
import { getHealth, startMatch, getJob } from './api.js'
import UploadScreen from './components/UploadScreen.jsx'
import TraceScreen from './components/TraceScreen.jsx'
import ResultsScreen from './components/ResultsScreen.jsx'

const STEPS = [
  { id: 'upload', num: 1, label: 'Upload' },
  { id: 'trace', num: 2, label: 'Trace' },
  { id: 'results', num: 3, label: 'Results' },
]

// Delay between the end of one status poll and the start of the next. Polls are
// chained (not fired on a fixed interval), so requests can never overlap.
const POLL_INTERVAL_MS = 1500

export default function App() {
  const [health, setHealth] = useState(null)
  const [screen, setScreen] = useState('upload')
  const [job, setJob] = useState(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState(null)
  const pollRef = useRef(null)
  const cancelledRef = useRef(false)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ gemini_key_present: false }))
  }, [])

  const stopPolling = useCallback(() => {
    cancelledRef.current = true
    clearTimeout(pollRef.current)
  }, [])

  // Poll the running job until it finishes, streaming records into the UI.
  // Each poll is scheduled only AFTER the previous request resolves (chained
  // setTimeout, not setInterval), so a slow backend can never accumulate a
  // backlog of overlapping in-flight requests.
  const poll = useCallback((jobId) => {
    clearTimeout(pollRef.current)
    cancelledRef.current = false
    const tick = async () => {
      try {
        const next = await getJob(jobId)
        if (cancelledRef.current) return
        setJob(next)
        if (next.status === 'running') {
          pollRef.current = setTimeout(tick, POLL_INTERVAL_MS)
        }
      } catch {
        // Stop polling on error; the last-known job state stays on screen.
      }
    }
    pollRef.current = setTimeout(tick, POLL_INTERVAL_MS)
  }, [])

  useEffect(() => () => stopPolling(), [stopPolling])

  const handleStart = async ({ inputsFile, masterFile, mode }) => {
    setStarting(true)
    setError(null)
    try {
      const { job_id } = await startMatch({ inputsFile, masterFile, mode })
      const initial = await getJob(job_id)
      setJob(initial)
      setScreen('trace')
      poll(job_id)
    } catch (e) {
      setError(e.message)
    } finally {
      setStarting(false)
    }
  }

  const handleReset = () => {
    stopPolling()
    setJob(null)
    setError(null)
    setScreen('upload')
  }

  // Patch one record in place after a human override (avoids clobbering it on
  // the next poll of a finished job — polling is stopped by then anyway).
  const patchRecord = (updated) => {
    setJob((prev) => {
      if (!prev) return prev
      return {
        ...prev,
        records: prev.records.map((r) => (r.record_id === updated.record_id ? updated : r)),
      }
    })
  }

  const hasResults = !!job && job.records.length > 0
  const runState = !job ? null : job.status === 'running' ? 'run' : job.status === 'error' ? 'error' : 'done'

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand__mark">MMV</span>
          <div>
            <div className="brand__name">Matcher</div>
          </div>
        </div>

        <nav className="stepper">
          {STEPS.map((s) => (
            <button
              key={s.id}
              className={`step${screen === s.id ? ' step--active' : ''}`}
              onClick={() => (s.id === 'upload' ? handleReset() : setScreen(s.id))}
              disabled={s.id !== 'upload' && !job}
            >
              <span className="step__num">{s.num}</span>
              {s.label}
            </button>
          ))}
        </nav>

        <div className="topbar__right">
          {job && (
            <span className={`mode-badge mode-badge--${job.mode === 'live' ? 'live' : 'mock'}`}>
              {job.mode === 'live' ? 'Live LLM' : 'Simulated'}
            </span>
          )}
          {runState && (
            <span className="run-pill">
              <span className={`dot dot--${runState}`} />
              {job.status === 'running'
                ? `${job.completed}/${job.total}`
                : job.status === 'error'
                ? 'error'
                : `${job.total} done`}
            </span>
          )}
        </div>
      </header>

      <main className="main">
        <div className="wrap">
          {screen === 'upload' && (
            <UploadScreen health={health} busy={starting} error={error} onStart={handleStart} />
          )}
          {screen === 'trace' && <TraceScreen job={job} />}
          {screen === 'results' && <ResultsScreen job={job} onRecordPatched={patchRecord} />}
        </div>
      </main>

      {job?.status === 'error' && screen !== 'upload' && (
        <div className="wrap" style={{ paddingTop: 0 }}>
          <div className="upload__err">Run failed: {job.error}</div>
        </div>
      )}

      {hasResults && screen === 'trace' && (
        <FloatingNext onClick={() => setScreen('results')} count={job.completed} status={job.status} />
      )}
    </div>
  )
}

function FloatingNext({ onClick, count, status }) {
  return (
    <div style={{ position: 'fixed', right: 28, bottom: 28, zIndex: 30 }}>
      <button className="btn btn--primary" onClick={onClick} style={{ boxShadow: 'var(--shadow)' }}>
        {status === 'running' ? `View results (${count} so far)` : 'View results table'} <span aria-hidden>→</span>
      </button>
    </div>
  )
}
