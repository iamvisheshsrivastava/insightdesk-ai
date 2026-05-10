import { useState } from 'react'

// Simple bar chart component
const SimpleBar = ({ title, xLabels, yValues }) => {
  const maxY = Math.max(...yValues, 1)
  return (
    <div style={{marginTop:12}}>
      <h4>{title}</h4>
      <div style={{display:'flex', gap:16, alignItems:'flex-end', height:200, justifyContent:'space-around'}}>
        {xLabels.map((label, i) => (
          <div key={i} style={{flex:1, textAlign:'center'}}>
            <div style={{
              height: (yValues[i] / maxY) * 150,
              backgroundColor: ['#1f77b4','#ff7f0e','#2ca02c'][i % 3],
              borderRadius:4,
              marginBottom:8
            }}></div>
            <div style={{fontSize:12, fontWeight:'bold'}}>{label}</div>
            <div style={{fontSize:11, color:'#666'}}>{(yValues[i] * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function TicketCategorization() {
  const [subject, setSubject] = useState('')
  const [description, setDescription] = useState('')
  const [errorLogs, setErrorLogs] = useState('')
  const [stackTrace, setStackTrace] = useState('')
  const [product, setProduct] = useState('Product A')
  const [priority, setPriority] = useState('Medium')
  const [severity, setSeverity] = useState('2 - Medium')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    const payload = { subject, description, error_logs: errorLogs, stack_trace: stackTrace, product, priority, severity }

    try {
      const res = await fetch('/api/predict/category', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <header className="page-header"><h1>📨 Ticket Categorization</h1><p>Classify support tickets</p></header>

      <div className="grid">
        <form className="card form-card" onSubmit={handleSubmit}>
          <label>Subject</label>
          <input value={subject} onChange={e => setSubject(e.target.value)} placeholder="Brief description" />

          <label>Description</label>
          <textarea value={description} onChange={e => setDescription(e.target.value)} rows={6} />

          <label>Error Logs</label>
          <textarea value={errorLogs} onChange={e => setErrorLogs(e.target.value)} rows={4} />

          <label>Stack Trace</label>
          <textarea value={stackTrace} onChange={e => setStackTrace(e.target.value)} rows={4} />

          <div className="two-col">
            <div>
              <label>Product</label>
              <select value={product} onChange={e => setProduct(e.target.value)}>
                <option>Product A</option>
                <option>Product B</option>
                <option>Product C</option>
                <option>Other</option>
              </select>
            </div>
            <div>
              <label>Priority</label>
              <select value={priority} onChange={e => setPriority(e.target.value)}>
                <option>Low</option>
                <option>Medium</option>
                <option>High</option>
                <option>Critical</option>
              </select>
            </div>
          </div>

          <label>Severity</label>
          <select value={severity} onChange={e => setSeverity(e.target.value)}>
            <option>1 - Low</option>
            <option>2 - Medium</option>
            <option>3 - High</option>
            <option>4 - Critical</option>
          </select>

          <div className="actions"><button type="submit" disabled={loading}>{loading ? 'Classifying...' : '🔍 Classify Ticket'}</button></div>
        </form>

        <aside className="card result-card">
          {error && <div className="error">API Error: {error}</div>}
          {!result && <div className="muted">Results will appear here after classification.</div>}
          {result && (
            <div>
              <h3>Results</h3>

              <div style={{display: 'flex', gap: 12}}>
                <div className="card" style={{flex:1}}>
                  <strong>Recommended:</strong>
                  <div>{(result.category) || result?.predictions?.xgboost?.category || 'Unknown'}</div>
                </div>
                <div className="card" style={{flex:1}}>
                  <strong>Inference Time:</strong>
                  <div>{(result.total_inference_time_ms ? `${result.total_inference_time_ms} ms` : 'N/A')}</div>
                </div>
                <div className="card" style={{flex:1}}>
                  <strong>Available Models:</strong>
                  <div>{(result.available_models || []).join(', ') || 'demo'}</div>
                </div>
              </div>

              {result?.predictions && (
                <SimpleBar 
                  title="Model Confidence Comparison"
                  xLabels={['XGBoost', 'TensorFlow']} 
                  yValues={[result?.predictions?.xgboost?.confidence || 0, result?.predictions?.tensorflow?.confidence || 0]} 
                />
              )}

              <details style={{marginTop:12}}>
                <summary>Raw JSON</summary>
                <pre className="json">{JSON.stringify(result, null, 2)}</pre>
              </details>
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}
