import { useState, useEffect } from 'react'

export default function Monitoring() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)

  async function fetchStatus() {
    setLoading(true)
    try {
      const res = await fetch('/api/monitoring/status')
      if (!res.ok) throw new Error(`${res.status}`)
      const data = await res.json()
      setStatus(data)
    } catch (e) { setStatus(null) } finally { setLoading(false) }
  }

  useEffect(() => { fetchStatus() }, [])

  return (
    <div>
      <header className="page-header"><h1>📊 Monitoring & Drift</h1><p>Model performance metrics</p></header>
      <div className="grid">
        <div className="card">
          <div className="actions small"><button onClick={fetchStatus}>🔄 Refresh</button></div>
          {loading && <div className="muted">Loading...</div>}
          {!loading && !status && <div className="muted">No monitoring data available.</div>}
          {status && (
            <div>
              <p><strong>Model Accuracy:</strong> {Math.round((status.accuracy||0)*100)/100}</p>
              <p><strong>Drift Score:</strong> {status.drift_score ?? 'N/A'}</p>
              <p><strong>Avg Latency:</strong> {status.avg_latency ?? 'N/A'} ms</p>
            </div>
          )}
        </div>

        <aside className="card result-card">
          <h3>System Health</h3>
          {status && status.system_health ? (
            <div>
              <p>CPU: {status.system_health.cpu_usage}%</p>
              <p>Memory: {status.system_health.memory_usage}%</p>
            </div>
          ) : <div className="muted">No system health data.</div>}
        </aside>
      </div>
    </div>
  )
}
