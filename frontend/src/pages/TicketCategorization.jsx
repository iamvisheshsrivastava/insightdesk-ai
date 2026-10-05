import { useState } from 'react'
import { Loader2, Tag, CheckCircle, Cpu, Zap } from 'lucide-react'
import { apiHeaders } from '../lib/apiHeaders'

const CATEGORY_COLORS = {
  'Technical Issue': 'bg-blue-100 text-blue-700',
  'Billing': 'bg-emerald-100 text-emerald-700',
  'Feature Request': 'bg-purple-100 text-purple-700',
  'Bug Report': 'bg-red-100 text-red-700',
  'Account': 'bg-amber-100 text-amber-700',
  'Security': 'bg-orange-100 text-orange-700',
  'Data Issue': 'bg-cyan-100 text-cyan-700',
  'Account Management': 'bg-amber-100 text-amber-700',
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
  // Must match TicketInput.validate_severity in src/api/main.py, which in turn matches
  // the training data vocab (data/support_tickets.json uses P0=highest .. P4=lowest).
  // Anything else is rejected and silently falls back to "P2" (#19).
  const [severity,    setSeverity]    = useState('P2')
  // Additional fields the model was actually trained on (see #16: the form previously
  // only sent 3 of 14 features, starving the classifier of its strongest signals).
  const [productModule,         setProductModule]         = useState('unknown')
  const [customerTier,          setCustomerTier]          = useState('standard')
  const [region,                setRegion]                = useState('US')
  const [previousTickets,       setPreviousTickets]       = useState(0)
  const [accountAgeDays,        setAccountAgeDays]        = useState(0)
  const [accountMonthlyValue,   setAccountMonthlyValue]   = useState(0)
  const [responseCount,         setResponseCount]         = useState(0)
  const [attachmentsCount,      setAttachmentsCount]      = useState(0)
  const [affectedUsers,         setAffectedUsers]         = useState(1)
  const [resolutionTimeHours,   setResolutionTimeHours]   = useState(0)
  const [loading,     setLoading]     = useState(false)
  const [result,      setResult]      = useState(null)
  const [error,       setError]       = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await fetch('/api/predict/category', {
        method: 'POST',
        headers: apiHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          subject, description, error_logs: errorLogs, stack_trace: stackTrace,
          product, priority, severity,
          product_module: productModule,
          customer_tier: customerTier,
          region,
          previous_tickets: Number(previousTickets) || 0,
          account_age_days: Number(accountAgeDays) || 0,
          account_monthly_value: Number(accountMonthlyValue) || 0,
          response_count: Number(responseCount) || 0,
          attachments_count: Number(attachmentsCount) || 0,
          affected_users: Number(affectedUsers) || 1,
          resolution_time_hours: Number(resolutionTimeHours) || 0,
        }),
      })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      setResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const preds = result?.predictions
  const primary = preds?.xgboost || preds?.tensorflow
  const category = primary?.predicted_category || primary?.category || result?.category
  const lowConfidence = primary?.low_confidence
  const suggestion = primary?.keyword_suggestion
  const colorClass = CATEGORY_COLORS[category] || 'bg-slate-100 text-slate-700'

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Page Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Tag size={20} className="text-blue-600" />
          <h1 className="text-xl font-bold text-slate-900">Ticket Classification</h1>
        </div>
        <p className="text-slate-500 text-sm">Classify support tickets with AI (XGBoost)</p>
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
                <option value="P0">P0 - Critical</option>
                <option value="P1">P1 - High</option>
                <option value="P2">P2 - Medium</option>
                <option value="P3">P3 - Low</option>
                <option value="P4">P4 - Lowest</option>
              </select>
            </div>
          </div>

          {/* Operational fields the model was trained on (#16) */}
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Product Module</label>
              <input
                value={productModule} onChange={e => setProductModule(e.target.value)}
                placeholder="e.g. billing-core"
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder:text-slate-400"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Customer Tier</label>
              <select value={customerTier} onChange={e => setCustomerTier(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option value="standard">Standard</option>
                <option value="premium">Premium</option>
                <option value="enterprise">Enterprise</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Region</label>
              <select value={region} onChange={e => setRegion(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option value="US">US</option>
                <option value="EU">EU</option>
                <option value="APAC">APAC</option>
                <option value="LATAM">LATAM</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Previous Tickets</label>
              <input
                type="number" min="0" value={previousTickets} onChange={e => setPreviousTickets(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Account Age (days)</label>
              <input
                type="number" min="0" value={accountAgeDays} onChange={e => setAccountAgeDays(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Account Value ($/mo)</label>
              <input
                type="number" min="0" step="0.01" value={accountMonthlyValue} onChange={e => setAccountMonthlyValue(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Affected Users</label>
              <input
                type="number" min="1" value={affectedUsers} onChange={e => setAffectedUsers(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Response Count</label>
              <input
                type="number" min="0" value={responseCount} onChange={e => setResponseCount(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Attachments</label>
              <input
                type="number" min="0" value={attachmentsCount} onChange={e => setAttachmentsCount(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Resolution Time (hrs)</label>
              <input
                type="number" min="0" step="0.1" value={resolutionTimeHours} onChange={e => setResolutionTimeHours(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
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
                {lowConfidence && (
                  <p className="mt-3 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-2">
                    Low model confidence.{suggestion ? ` Keyword analysis suggests: ${suggestion}.` : ' Treat this result as a guess.'}
                  </p>
                )}
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
                      XGBoost → <span className="font-medium text-slate-700">{(result.predictions.xgboost?.predicted_category || result.predictions.xgboost?.category)}</span>
                    </p>
                    <p className="text-xs text-slate-500">
                      TensorFlow → <span className="font-medium text-slate-700">{(result.predictions.tensorflow?.predicted_category || result.predictions.tensorflow?.category)}</span>
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
