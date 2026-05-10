import { useState } from 'react'

export default function SolutionRetrieval() {
  const [subject, setSubject] = useState('')
  const [description, setDescription] = useState('')
  const [topK, setTopK] = useState(5)
  const [searchType, setSearchType] = useState('hybrid')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  async function handleSearch(e) {
    e.preventDefault()
    setLoading(true); setError(null); setResults(null)
    const payload = { subject: subject || 'General Issue', description: description || '', k: topK, search_type: searchType }
    try {
      const res = await fetch('/api/retrieve/solutions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const data = await res.json()
      setResults(data)
    } catch (err) { setError(err.message) } finally { setLoading(false) }
  }

  return (
    <div>
      <header className="page-header"><h1>🔎 Solution Retrieval</h1><p>Find relevant solutions using RAG</p></header>

      <div className="grid">
        <form className="card form-card" onSubmit={handleSearch}>
          <label>Subject</label>
          <input value={subject} onChange={e => setSubject(e.target.value)} />

          <label>Description</label>
          <textarea value={description} onChange={e => setDescription(e.target.value)} rows={6} />

          <div className="two-col">
            <div>
              <label>Number of Solutions</label>
              <input type="number" min={1} max={10} value={topK} onChange={e => setTopK(Number(e.target.value))} />
            </div>
            <div>
              <label>Search Type</label>
              <select value={searchType} onChange={e => setSearchType(e.target.value)}>
                <option value="hybrid">hybrid</option>
                <option value="semantic">semantic</option>
                <option value="keyword">keyword</option>
              </select>
            </div>
          </div>

          <div className="actions"><button type="submit" disabled={loading}>{loading ? 'Searching...' : '🔍 Find Solutions'}</button></div>
        </form>

        <aside className="card result-card">
          {error && <div className="error">API Error: {error}</div>}
          {!results && <div className="muted">Search results will appear here.</div>}
          {results && results.solutions && (
            <div>
              <h3>Found {results.solutions.length} solutions</h3>
              <ul className="solutions-list">
                {results.solutions.map((s, i) => (
                  <li key={i} className="solution-item">
                    <strong>#{i+1}</strong> <em>Score: {s.score?.toFixed(3) ?? 'N/A'}</em>
                    <p>{s.resolution || s.preview || 'No resolution text'}</p>
                    {s.ticket_id && <small>Ticket: {s.ticket_id}</small>}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}
