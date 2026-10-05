import { useState, useEffect } from 'react'
import { PieChart, RefreshCw, Loader2, Activity, Gauge, Layers } from 'lucide-react'

const BAR_COLORS = [
  'bg-blue-500', 'bg-purple-500', 'bg-emerald-500', 'bg-amber-500',
  'bg-rose-500', 'bg-cyan-500', 'bg-indigo-500', 'bg-pink-500',
]

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

function BreakdownBars({ title, data }) {
  const entries = Object.entries(data || {}).sort((a, b) => b[1] - a[1])
  const max = entries.length ? entries[0][1] : 0
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
      <p className="text-sm font-semibold text-slate-700 mb-4">{title}</p>
      {entries.length === 0 && <p className="text-sm text-slate-400">No data in this window.</p>}
      <div className="space-y-3">
        {entries.map(([label, count], i) => (
          <div key={label}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-600 font-medium">{label}</span>
              <span className="font-semibold text-slate-800">{count}</span>
            </div>
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${BAR_COLORS[i % BAR_COLORS.length]}`}
                style={{ width: max ? `${Math.max((count / max) * 100, 3)}%` : '0%' }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function VolumeSparkline({ volumeByDay }) {
  const max = volumeByDay.reduce((m, d) => Math.max(m, d.count), 0) || 1
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
      <p className="text-sm font-semibold text-slate-700 mb-4">Ticket Volume Over Time</p>
      {volumeByDay.length === 0 ? (
        <p className="text-sm text-slate-400">No data in this window.</p>
      ) : (
        <div className="flex items-end gap-1 h-32">
          {volumeByDay.map((d) => (
            <div key={d.day} className="flex-1 flex flex-col items-center justify-end group relative">
              <div
                className="w-full bg-blue-500 rounded-t-sm hover:bg-blue-600 transition-colors"
                style={{ height: `${Math.max((d.count / max) * 100, 2)}%` }}
                title={`${d.day}: ${d.count} tickets`}
              />
            </div>
          ))}
        </div>
      )}
      {volumeByDay.length > 0 && (
        <div className="flex justify-between text-[10px] text-slate-400 mt-2">
          <span>{volumeByDay[0].day}</span>
          <span>{volumeByDay[volumeByDay.length - 1].day}</span>
        </div>
      )}
    </div>
  )
}

export default function Analytics() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [days,    setDays]    = useState(30)

  async function fetchSummary() {
    setLoading(true)
    try {
      const res = await fetch(`/api/analytics/summary?days=${days}`)
      if (!res.ok) throw new Error(`${res.status}`)
      setData(await res.json())
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchSummary() }, [days])

  const confidencePct = data?.avg_model_confidence != null ? Math.round(data.avg_model_confidence * 100) : null

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <PieChart size={20} className="text-blue-600" />
            <h1 className="text-xl font-bold text-slate-900">Analytics</h1>
          </div>
          <p className="text-slate-500 text-sm">Ticket volume and category trends</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={days} onChange={e => setDays(Number(e.target.value))}
            className="text-sm border border-slate-200 rounded-lg px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
          <button
            onClick={fetchSummary} disabled={loading}
            className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900 bg-white border border-slate-200 px-3 py-2 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="bg-white rounded-xl border border-slate-200 p-10 text-center">
          <Loader2 size={24} className="animate-spin text-blue-500 mx-auto mb-3" />
          <p className="text-slate-500 text-sm">Loading analytics...</p>
        </div>
      )}

      {/* Empty */}
      {!loading && !data && (
        <div className="bg-white rounded-xl border border-slate-200 p-10 text-center">
          <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-3">
            <Activity size={20} className="text-slate-400" />
          </div>
          <p className="text-slate-500 text-sm">Could not load analytics data.</p>
        </div>
      )}

      {!loading && data && (
        <div className="space-y-6">
          {data.status === 'no_data' && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-700">
              No ticket data available to summarize.
            </div>
          )}

          {/* KPIs */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <KpiCard icon={Layers} label="Tickets in Window" value={data.total_tickets} color="bg-blue-500" />
            <KpiCard
              icon={Gauge} label="Avg Model Confidence"
              value={confidencePct != null ? confidencePct : '—'}
              unit={confidencePct != null ? '%' : ''}
              color="bg-purple-500"
            />
            <KpiCard icon={Activity} label="Corrections Logged" value={data.corrections_in_window} color="bg-emerald-500" />
            <KpiCard icon={PieChart} label="Window (days)" value={data.window_days} color="bg-amber-500" />
          </div>

          <VolumeSparkline volumeByDay={data.volume_by_day || []} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <BreakdownBars title="By Category" data={data.category_breakdown} />
            <BreakdownBars title="By Priority" data={data.priority_breakdown} />
          </div>
        </div>
      )}
    </div>
  )
}
