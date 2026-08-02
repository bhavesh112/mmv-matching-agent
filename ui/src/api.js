// Thin client for the MMV matching API (server.py). Same-origin in production;
// proxied through Vite to :8000 in dev.

async function jsonOrThrow(res) {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body && body.detail) detail = body.detail
    } catch {
      /* non-JSON error body */
    }
    const err = new Error(detail)
    // Expose the HTTP status so callers can tell transient gateway failures
    // (502/503/504) from terminal ones (e.g. 404). Network-level failures
    // reject before reaching here and carry no status at all.
    err.status = res.status
    throw err
  }
  return res.json()
}

export async function getHealth() {
  return jsonOrThrow(await fetch('/api/health'))
}

export async function getSample() {
  return jsonOrThrow(await fetch('/api/sample'))
}

export async function startMatch({ inputsFile, masterFile, mode }) {
  const form = new FormData()
  form.append('inputs', inputsFile)
  if (masterFile) form.append('master', masterFile)
  form.append('mode', mode)
  return jsonOrThrow(await fetch('/api/match', { method: 'POST', body: form }))
}

export async function getJob(jobId) {
  return jsonOrThrow(await fetch(`/api/jobs/${jobId}`))
}

export async function overrideRecord(jobId, recordId, decision, note) {
  return jsonOrThrow(
    await fetch(`/api/jobs/${jobId}/records/${recordId}/override`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ decision, note }),
    }),
  )
}
