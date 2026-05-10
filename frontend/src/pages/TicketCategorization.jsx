import { useState } from 'react'
import Plot from 'react-plotly.js'

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

              {/* Summary metrics */}
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

              {/* Model confidence comparison chart */}
              <div style={{marginTop:12}}>
                <h4>Model Confidence Comparison</h4>
                <Plot
                  data={[
                    {
                      x: ['XGBoost','TensorFlow'],
                      y: [result?.predictions?.xgboost?.confidence || 0, result?.predictions?.tensorflow?.confidence || 0],
                      type: 'bar', marker: {color: ['#1f77b4','#ff7f0e']}
                    }
                  ]}
                  layout={{width: '100%', height: 300, margin: {t:20}}}
                  useResizeHandler
                  style={{width: '100%'}}
                />
              </div>

              {/* Probability distribution (if available) */}
              {result.probabilities && (
                <div style={{marginTop:12}}>
                  <h4>Top Category Probabilities</h4>
                  <Plot
                    data={[{
                      x: Object.keys(result.probabilities),
                      y: Object.values(result.probabilities),
                      type: 'bar'
                    }]}
                    layout={{width:'100%', height:300, margin:{t:20}}}
                    useResizeHandler
                    style={{width:'100%'}}
                  />
                </div>
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
