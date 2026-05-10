import { useState, useEffect } from 'react'

export default function Anomalies() {
  const [timeRange, setTimeRange] = useState('Last 24 Hours')
  const [severity, setSeverity] = useState('All')
  const [anomalies, setAnomalies] = useState(null)
  const [loading, setLoading] = useState(false)

  async function fetchAnomalies() {
    setLoading(true)
    try {
      const res = await fetch('/api/anomalies/recent')
      if (!res.ok) throw new Error(`${res.status}`)
      const data = await res.json()
      setAnomalies(data.anomalies || [])
    } catch (e) {
      setAnomalies([])
    } finally { setLoading(false) }
  }

  useEffect(() => { fetchAnomalies() }, [])

  const filtered = (anomalies || []).filter(a => severity === 'All' ? true : (a.severity || '').toLowerCase() === severity.toLowerCase())

  return (
    <div>
      <header className="page-header"><h1>🚨 Anomaly Detection</h1><p>Recent system anomalies</p></header>

      <div className="controls two-col">
        <div>
          <label>Time Range</label>
          <select value={timeRange} onChange={e => setTimeRange(e.target.value)}>
            <option>Last Hour</option>
            <option>Last 6 Hours</option>
            <option>Last 24 Hours</option>
            <option>Last Week</option>
          </select>
        </div>
        <div>
          <label>Severity Filter</label>
          <select value={severity} onChange={e => setSeverity(e.target.value)}>
            <option>All</option>
            <option>High</option>
            <option>Medium</option>
            <option>Low</option>
          </select>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <div className="actions small"><button onClick={fetchAnomalies}>🔄 Refresh</button></div>
          {loading && <div className="muted">Loading anomalies...</div>}
          {!loading && (!filtered || filtered.length === 0) && <div className="muted">No anomalies found.</div>}

          {!loading && filtered && filtered.map((a, i) => (
            <div key={i} className={`anomaly ${a.severity?.toLowerCase()}`}>
              <strong>{a.type || 'Unknown'}</strong>
              <div>{a.details || ''}</div>
              <small>{a.timestamp}</small>
            </div>
          ))}
        </div>

        <aside className="card result-card">
          <h3>Summary</h3>
          <p>Total: {filtered.length}</p>
        </aside>
      </div>
    </div>
  )
}
