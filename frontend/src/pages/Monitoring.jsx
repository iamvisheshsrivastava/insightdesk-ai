import { useState, useEffect } from 'react'
import { BarChart2, RefreshCw, Loader2, Cpu, Gauge, TrendingUp, Activity } from 'lucide-react'

function KpiCard({ icon: Icon, label, value, unit, color }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
      <div className={`w-9 h-9 rounded-lg flex items-center justify-center mb-3 ${color}`}>
        <Icon size={18} className="text-white" />
      </div>
      <p className="text-2xl font-bold text-slate-900">
        {value != null ? `${value}${unit || ''}` : '—'}
      </p>
      <p className="text-xs font-medium text-slate-500 mt-0.5">{label}</p>
    </div>
  )
}

function ProgressBar({ label, value, max = 100, color }) {
  const pct = Math.min(Math.round((value / max) * 100), 100)
  return (
    <div>
      <div className="flex justify-between text-xs mb-1.5">
        <span className="text-slate-600 font-medium">{label}</span>
        <span className="font-semibold text-slate-800">{Math.round(value ?? 0)}{max === 100 ? '%' : ''}</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-700 ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export default function Monitoring() {
  const [status,  setStatus]  = useState(null)
  const [loading, setLoading] = useState(false)

  async function fetchStatus() {
    setLoading(true)
    try {
      const res = await fetch('/api/monitoring/status')
      if (!res.ok) throw new Error(`${res.status}`)
      setStatus(await res.json())
    } catch {
      setStatus(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStatus() }, [])

  const acc     = status ? Math.round((status.accuracy || 0) * 100) : null
  const f1      = status ? Math.round((status.weighted_f1 || 0) * 100) : null
  const drift   = status ? (status.drift_score?.toFixed(3) ?? null) : null
  const latency = status ? Math.round(status.avg_latency ?? 0) : null

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <BarChart2 size={20} className="text-blue-600" />
            <h1 className="text-xl font-bold text-slate-900">Monitoring & Drift</h1>
          </div>
          <p className="text-slate-500 text-sm">Model performance metrics and system health</p>
        </div>
        <button
          onClick={fetchStatus} disabled={loading}
          className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900 bg-white border border-slate-200 px-3 py-2 rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="bg-white rounded-xl border border-slate-200 p-10 text-center">
          <Loader2 size={24} className="animate-spin text-blue-500 mx-auto mb-3" />
          <p className="text-slate-500 text-sm">Loading monitoring data...</p>
        </div>
      )}

      {/* Empty */}
      {!loading && !status && (
        <div className="bg-white rounded-xl border border-slate-200 p-10 text-center">
          <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-3">
            <Activity size={20} className="text-slate-400" />
          </div>
          <p className="text-slate-500 text-sm">No monitoring data available. Backend may be offline.</p>
        </div>
      )}

      {!loading && status && (
        <div className="space-y-6">
          {/* KPI Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <KpiCard icon={TrendingUp} label="Model Accuracy"  value={acc}     unit="%" color="bg-blue-600" />
            <KpiCard icon={Activity}   label="Weighted F1"     value={f1}      unit="%" color="bg-purple-600" />
            <KpiCard icon={Gauge}      label="Drift Score"     value={drift}   unit=""  color="bg-amber-500" />
            <KpiCard icon={Cpu}        label="Avg Latency"     value={latency} unit=" ms" color="bg-emerald-600" />
          </div>

          {/* Model Comparison */}
          {status.model_comparison && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <h2 className="text-sm font-semibold text-slate-900 mb-4">Model Comparison</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-100">
                      <th className="text-left text-xs font-semibold text-slate-500 uppercase tracking-wide pb-3">Model</th>
                      <th className="text-right text-xs font-semibold text-slate-500 uppercase tracking-wide pb-3">Accuracy</th>
                      <th className="text-right text-xs font-semibold text-slate-500 uppercase tracking-wide pb-3">F1</th>
                      <th className="text-right text-xs font-semibold text-slate-500 uppercase tracking-wide pb-3">Latency (ms)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-50">
                    {status.model_comparison.map((m, i) => (
                      <tr key={i} className="hover:bg-slate-50">
                        <td className="py-3 font-medium text-slate-900">{m.Model}</td>
                        <td className="py-3 text-right text-slate-700">{(m.Accuracy * 100).toFixed(1)}%</td>
                        <td className="py-3 text-right text-slate-700">{(m.F1 * 100).toFixed(1)}%</td>
                        <td className="py-3 text-right text-slate-700">{Math.round(m.Latency)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* System Health */}
          {status.system_health && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <h2 className="text-sm font-semibold text-slate-900 mb-4">System Health</h2>
              <div className="space-y-4">
                <ProgressBar label="CPU Usage"    value={status.system_health.cpu_usage}    color="bg-blue-500" />
                <ProgressBar label="Memory Usage" value={status.system_health.memory_usage} color="bg-purple-500" />
                {status.system_health.requests_per_minute != null && (
                  <div className="pt-2 border-t border-slate-100 flex items-center justify-between">
                    <span className="text-sm text-slate-600 font-medium">Requests / min</span>
                    <span className="text-sm font-bold text-slate-900">{Math.round(status.system_health.requests_per_minute)}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
