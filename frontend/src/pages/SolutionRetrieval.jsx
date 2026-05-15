import { useState } from 'react'
import { Search, Loader2, FileText, TrendingUp } from 'lucide-react'

function ScoreBadge({ score }) {
  const pct = score != null ? Math.round(score * 100) : null
  const cls = pct >= 80 ? 'bg-emerald-100 text-emerald-700' :
              pct >= 50 ? 'bg-amber-100 text-amber-700' :
                          'bg-red-100 text-red-700'
  return pct != null
    ? <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${cls}`}>{pct}% match</span>
    : <span className="text-xs text-slate-400">N/A</span>
}

export default function SolutionRetrieval() {
  const [subject,     setSubject]     = useState('')
  const [description, setDescription] = useState('')
  const [topK,        setTopK]        = useState(5)
  const [searchType,  setSearchType]  = useState('hybrid')
  const [loading,     setLoading]     = useState(false)
  const [results,     setResults]     = useState(null)
  const [error,       setError]       = useState(null)

  async function handleSearch(e) {
    e.preventDefault()
    setLoading(true); setError(null); setResults(null)
    try {
      const res = await fetch('/api/retrieve/solutions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject: subject || 'General Issue', description, k: topK, search_type: searchType }),
      })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      setResults(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const solutions = results?.solutions || []

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Page Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Search size={20} className="text-blue-600" />
          <h1 className="text-xl font-bold text-slate-900">Solution Retrieval</h1>
        </div>
        <p className="text-slate-500 text-sm">Find relevant solutions using RAG-based semantic search</p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Subject</label>
            <input
              value={subject} onChange={e => setSubject(e.target.value)}
              placeholder="e.g. login not working"
              className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder:text-slate-400"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Results</label>
              <input
                type="number" min={1} max={10} value={topK}
                onChange={e => setTopK(Number(e.target.value))}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Search Type</label>
              <select value={searchType} onChange={e => setSearchType(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option value="hybrid">Hybrid</option>
                <option value="semantic">Semantic</option>
                <option value="keyword">Keyword</option>
              </select>
            </div>
          </div>
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-1.5">Description</label>
          <textarea
            value={description} onChange={e => setDescription(e.target.value)}
            rows={3} placeholder="Describe the issue in detail..."
            className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder:text-slate-400 resize-none"
          />
        </div>

        <button
          type="submit" disabled={loading}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white text-sm font-semibold px-5 py-2.5 rounded-lg transition-colors"
        >
          {loading ? <><Loader2 size={15} className="animate-spin" /> Searching...</> : <><Search size={15} /> Find Solutions</>}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700 mb-4">
          <strong>API Error:</strong> {error}
        </div>
      )}

      {/* Empty State */}
      {!loading && !results && !error && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-10 text-center">
          <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-3">
            <FileText size={20} className="text-slate-400" />
          </div>
          <p className="text-slate-500 text-sm">Search results will appear here.</p>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-10 text-center">
          <Loader2 size={24} className="animate-spin text-blue-500 mx-auto mb-3" />
          <p className="text-slate-500 text-sm">Searching knowledge base...</p>
        </div>
      )}

      {/* Results */}
      {!loading && results && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm text-slate-600 mb-1">
            <TrendingUp size={15} className="text-blue-500" />
            Found <strong>{solutions.length}</strong> solutions
          </div>

          {solutions.length === 0 && (
            <div className="bg-white rounded-xl border border-slate-200 p-6 text-center text-slate-500 text-sm">
              No solutions found. Try a different query or search type.
            </div>
          )}

          {solutions.map((s, i) => (
            <div key={i} className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center text-xs font-bold shrink-0">
                    {i + 1}
                  </span>
                  {s.ticket_id && (
                    <span className="text-xs text-slate-500 font-mono">Ticket: {s.ticket_id}</span>
                  )}
                </div>
                <ScoreBadge score={s.score} />
              </div>
              <p className="text-sm text-slate-700 leading-relaxed">
                {s.resolution || s.preview || 'No resolution text available.'}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
