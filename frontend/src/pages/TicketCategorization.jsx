import { useState } from 'react'
import { Loader2, Tag, CheckCircle, Cpu, Zap } from 'lucide-react'

const CATEGORIES = ['Technical Issue', 'Billing', 'Feature Request', 'Bug Report', 'Account', 'Other']
const CATEGORY_COLORS = {
  'Technical Issue': 'bg-blue-100 text-blue-700',
  'Billing': 'bg-emerald-100 text-emerald-700',
  'Feature Request': 'bg-purple-100 text-purple-700',
  'Bug Report': 'bg-red-100 text-red-700',
  'Account': 'bg-amber-100 text-amber-700',
  'Other': 'bg-slate-100 text-slate-700',
}

function ConfidenceBar({ label, value, color }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-600 font-medium">{label}</span>
        <span className="font-semibold text-slate-800">{pct}%</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export default function TicketCategorization() {
  const [subject,     setSubject]     = useState('')
  const [description, setDescription] = useState('')
  const [errorLogs,   setErrorLogs]   = useState('')
  const [stackTrace,  setStackTrace]  = useState('')
  const [product,     setProduct]     = useState('Product A')
  const [priority,    setPriority]    = useState('Medium')
  const [severity,    setSeverity]    = useState('2 - Medium')
  const [loading,     setLoading]     = useState(false)
  const [result,      setResult]      = useState(null)
  const [error,       setError]       = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await fetch('/api/predict/category', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject, description, error_logs: errorLogs, stack_trace: stackTrace, product, priority, severity }),
      })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      setResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const category = result?.category || result?.predictions?.xgboost?.category
  const colorClass = CATEGORY_COLORS[category] || 'bg-slate-100 text-slate-700'

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Page Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Tag size={20} className="text-blue-600" />
          <h1 className="text-xl font-bold text-slate-900">Ticket Classification</h1>
        </div>
        <p className="text-slate-500 text-sm">Classify support tickets with dual-model AI (XGBoost + TensorFlow)</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Form */}
        <form onSubmit={handleSubmit} className="lg:col-span-3 bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Subject</label>
            <input
              value={subject} onChange={e => setSubject(e.target.value)}
              placeholder="Brief description of the issue"
              className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder:text-slate-400"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Description</label>
            <textarea
              value={description} onChange={e => setDescription(e.target.value)}
              rows={5} placeholder="Detailed description of the issue..."
              className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder:text-slate-400 resize-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Error Logs</label>
            <textarea
              value={errorLogs} onChange={e => setErrorLogs(e.target.value)}
              rows={3} placeholder="Paste error logs here..."
              className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder:text-slate-400 resize-none font-mono"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Stack Trace</label>
            <textarea
              value={stackTrace} onChange={e => setStackTrace(e.target.value)}
              rows={3} placeholder="Paste stack trace here..."
              className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder:text-slate-400 resize-none font-mono"
            />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Product</label>
              <select value={product} onChange={e => setProduct(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option>Product A</option>
                <option>Product B</option>
                <option>Product C</option>
                <option>Other</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Priority</label>
              <select value={priority} onChange={e => setPriority(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option>Low</option>
                <option>Medium</option>
                <option>High</option>
                <option>Critical</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Severity</label>
              <select value={severity} onChange={e => setSeverity(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option>1 - Low</option>
                <option>2 - Medium</option>
                <option>3 - High</option>
                <option>4 - Critical</option>
              </select>
            </div>
          </div>

          <button
            type="submit" disabled={loading}
            className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white text-sm font-semibold py-2.5 rounded-lg transition-colors"
          >
            {loading ? <><Loader2 size={15} className="animate-spin" /> Classifying...</> : 'Classify Ticket'}
          </button>
        </form>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700">
              <strong>API Error:</strong> {error}
            </div>
          )}

          {!result && !error && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 text-center">
              <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <Tag size={20} className="text-slate-400" />
              </div>
              <p className="text-slate-500 text-sm">Submit a ticket to see AI predictions here.</p>
            </div>
          )}

          {result && (
            <>
              {/* Category Result */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle size={16} className="text-emerald-500" />
                  <span className="text-sm font-semibold text-slate-700">Prediction Result</span>
                </div>
                <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${colorClass}`}>
                  {category || 'Unknown'}
                </span>
                {result.total_inference_time_ms && (
                  <div className="flex items-center gap-1.5 mt-3 text-xs text-slate-500">
                    <Zap size={12} />
                    Inference: {result.total_inference_time_ms} ms
                  </div>
                )}
              </div>

              {/* Model Comparison */}
              {result.predictions && (
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
                  <div className="flex items-center gap-2 mb-4">
                    <Cpu size={16} className="text-blue-500" />
                    <span className="text-sm font-semibold text-slate-700">Model Confidence</span>
                  </div>
                  <div className="space-y-3">
                    <ConfidenceBar
                      label="XGBoost"
                      value={result.predictions.xgboost?.confidence}
                      color="bg-blue-500"
                    />
                    <ConfidenceBar
                      label="TensorFlow"
                      value={result.predictions.tensorflow?.confidence}
                      color="bg-purple-500"
                    />
                  </div>
                  <div className="mt-3 pt-3 border-t border-slate-100 space-y-1">
                    <p className="text-xs text-slate-500">
                      XGBoost → <span className="font-medium text-slate-700">{result.predictions.xgboost?.category}</span>
                    </p>
                    <p className="text-xs text-slate-500">
                      TensorFlow → <span className="font-medium text-slate-700">{result.predictions.tensorflow?.category}</span>
                    </p>
                  </div>
                </div>
              )}

              {/* Raw JSON */}
              <details className="bg-slate-900 rounded-xl overflow-hidden">
                <summary className="px-4 py-2.5 text-xs font-medium text-slate-400 cursor-pointer hover:text-slate-300">
                  Raw JSON Response
                </summary>
                <pre className="px-4 pb-4 text-xs text-emerald-300 overflow-auto max-h-48">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
