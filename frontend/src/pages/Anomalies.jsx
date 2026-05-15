import { useState, useEffect } from 'react'
import { AlertTriangle, RefreshCw, Loader2, ShieldAlert, Clock } from 'lucide-react'

const SEVERITY_STYLES = {
  high:   { badge: 'bg-red-100 text-red-700 border-red-200',   dot: 'bg-red-500',   border: 'border-l-red-500' },
  medium: { badge: 'bg-amber-100 text-amber-700 border-amber-200', dot: 'bg-amber-500', border: 'border-l-amber-500' },
  low:    { badge: 'bg-blue-100 text-blue-700 border-blue-200',  dot: 'bg-blue-400',  border: 'border-l-blue-400' },
}

function SeverityBadge({ severity }) {
  const s = (severity || 'low').toLowerCase()
  const styles = SEVERITY_STYLES[s] || SEVERITY_STYLES.low
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2 py-0.5 rounded-full border ${styles.badge}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${styles.dot}`} />
      {severity || 'Low'}
    </span>
  )
}

function StatChip({ label, count, color }) {
  return (
    <div className={`bg-white rounded-xl border border-slate-200 shadow-sm px-5 py-4`}>
      <p className="text-2xl font-bold text-slate-900">{count}</p>
      <p className={`text-xs font-medium mt-0.5 ${color}`}>{label}</p>
    </div>
  )
}

export default function Anomalies() {
  const [severity,  setSeverity]  = useState('All')
  const [anomalies, setAnomalies] = useState(null)
  const [loading,   setLoading]   = useState(false)

  async function fetchAnomalies() {
    setLoading(true)
    try {
      const res = await fetch('/api/anomalies/recent')
      if (!res.ok) throw new Error(`${res.status}`)
      const data = await res.json()
      setAnomalies(data.anomalies || [])
    } catch {
      setAnomalies([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchAnomalies() }, [])

  const all = anomalies || []
  const high   = all.filter(a => a.severity?.toLowerCase() === 'high').length
  const medium = all.filter(a => a.severity?.toLowerCase() === 'medium').length
  const low    = all.filter(a => a.severity?.toLowerCase() === 'low').length

  const filtered = severity === 'All' ? all : all.filter(a =>
    a.severity?.toLowerCase() === severity.toLowerCase()
  )

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle size={20} className="text-amber-500" />
            <h1 className="text-xl font-bold text-slate-900">Anomaly Detection</h1>
          </div>
          <p className="text-slate-500 text-sm">Real-time system anomalies and alerts</p>
        </div>
        <button
          onClick={fetchAnomalies} disabled={loading}
          className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900 bg-white border border-slate-200 px-3 py-2 rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <StatChip label="Total" count={all.length} color="text-slate-500" />
        <StatChip label="High" count={high} color="text-red-600" />
        <StatChip label="Medium" count={medium} color="text-amber-600" />
        <StatChip label="Low" count={low} color="text-blue-600" />
      </div>

      {/* Filter */}
      <div className="flex items-center gap-3 mb-4">
        <span className="text-sm font-medium text-slate-600">Filter by severity:</span>
        {['All', 'High', 'Medium', 'Low'].map(s => (
          <button
            key={s} onClick={() => setSeverity(s)}
            className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-colors ${
              severity === s
                ? 'bg-slate-900 text-white border-slate-900'
                : 'bg-white text-slate-600 border-slate-200 hover:border-slate-400'
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      {/* List */}
      {loading && (
        <div className="bg-white rounded-xl border border-slate-200 p-10 text-center">
          <Loader2 size={24} className="animate-spin text-blue-500 mx-auto mb-3" />
          <p className="text-slate-500 text-sm">Loading anomalies...</p>
        </div>
      )}

      {!loading && filtered.length === 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-10 text-center">
          <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-3">
            <ShieldAlert size={20} className="text-slate-400" />
          </div>
          <p className="text-slate-500 text-sm">No anomalies found{severity !== 'All' ? ` for severity: ${severity}` : ''}.</p>
        </div>
      )}

      {!loading && filtered.length > 0 && (
        <div className="space-y-3">
          {filtered.map((a, i) => {
            const s = (a.severity || 'low').toLowerCase()
            const styles = SEVERITY_STYLES[s] || SEVERITY_STYLES.low
            return (
              <div key={i} className={`bg-white rounded-xl border border-slate-200 border-l-4 shadow-sm p-5 ${styles.border}`}>
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1.5">
                      <SeverityBadge severity={a.severity} />
                      <span className="font-semibold text-slate-900 text-sm">{a.type || 'Unknown'}</span>
                    </div>
                    <p className="text-sm text-slate-600 leading-relaxed">{a.details || ''}</p>
                  </div>
                  {a.timestamp && (
                    <div className="flex items-center gap-1 text-xs text-slate-400 shrink-0">
                      <Clock size={11} />
                      {new Date(a.timestamp).toLocaleString()}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
