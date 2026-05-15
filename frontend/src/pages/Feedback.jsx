import { useState } from 'react'
import { MessageSquare, Loader2, CheckCircle, Star } from 'lucide-react'

function StarRating({ value, onChange }) {
  const [hovered, setHovered] = useState(0)
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map(n => (
        <button
          key={n} type="button"
          onClick={() => onChange(n)}
          onMouseEnter={() => setHovered(n)}
          onMouseLeave={() => setHovered(0)}
          className="p-0.5"
        >
          <Star
            size={22}
            className={`transition-colors ${n <= (hovered || value) ? 'text-amber-400 fill-amber-400' : 'text-slate-300'}`}
          />
        </button>
      ))}
    </div>
  )
}

export default function Feedback() {
  const [ticketId,         setTicketId]         = useState('')
  const [predCategory,     setPredCategory]     = useState('')
  const [actualCategory,   setActualCategory]   = useState('Technical Issue')
  const [predictionCorrect,setPredictionCorrect]= useState('Yes')
  const [resolutionTime,   setResolutionTime]   = useState(1.0)
  const [agentSatisfaction,setAgentSatisfaction]= useState(3)
  const [notes,            setNotes]            = useState('')
  const [loading,          setLoading]          = useState(false)
  const [success,          setSuccess]          = useState(false)
  const [error,            setError]            = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true); setError(null); setSuccess(false)
    try {
      const res = await fetch('/api/feedback/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticket_id: ticketId, predicted_category: predCategory,
          actual_category: actualCategory, prediction_correct: predictionCorrect,
          resolution_time: resolutionTime, agent_satisfaction: agentSatisfaction,
          notes, feedback_type: 'agent',
        }),
      })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      setSuccess(true)
      setTicketId(''); setPredCategory(''); setNotes('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* Page Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <MessageSquare size={20} className="text-blue-600" />
          <h1 className="text-xl font-bold text-slate-900">Feedback</h1>
        </div>
        <p className="text-slate-500 text-sm">Submit agent feedback to improve model performance over time</p>
      </div>

      {/* Success Banner */}
      {success && (
        <div className="flex items-center gap-3 bg-emerald-50 border border-emerald-200 rounded-xl p-4 mb-5">
          <CheckCircle size={18} className="text-emerald-600 shrink-0" />
          <div>
            <p className="text-sm font-semibold text-emerald-800">Feedback submitted successfully</p>
            <p className="text-xs text-emerald-600 mt-0.5">Thank you — your input helps improve AI predictions.</p>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700 mb-5">
          <strong>Submission failed:</strong> {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form */}
        <form onSubmit={handleSubmit} className="lg:col-span-2 bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-5">
          {/* Ticket Info */}
          <div>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Ticket Information</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">Ticket ID</label>
                <input value={ticketId} onChange={e => setTicketId(e.target.value)}
                  placeholder="TKT-12345"
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder:text-slate-400 font-mono"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">Predicted Category</label>
                <input value={predCategory} onChange={e => setPredCategory(e.target.value)}
                  placeholder="e.g. Technical Issue"
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder:text-slate-400"
                />
              </div>
            </div>
          </div>

          <div className="border-t border-slate-100" />

          {/* Prediction Quality */}
          <div>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Prediction Quality</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">Actual Category</label>
                <select value={actualCategory} onChange={e => setActualCategory(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                  {['Technical Issue', 'Billing', 'Feature Request', 'Bug Report', 'Account', 'Other'].map(c => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">Was Prediction Correct?</label>
                <div className="flex gap-2">
                  {['Yes', 'No', 'Partially'].map(opt => (
                    <button key={opt} type="button"
                      onClick={() => setPredictionCorrect(opt)}
                      className={`flex-1 text-sm py-2 rounded-lg font-medium border transition-colors ${
                        predictionCorrect === opt
                          ? 'bg-blue-600 text-white border-blue-600'
                          : 'bg-white text-slate-600 border-slate-200 hover:border-slate-400'
                      }`}
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="border-t border-slate-100" />

          {/* Performance */}
          <div>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Performance Metrics</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">Resolution Time (hours)</label>
                <input type="number" value={resolutionTime}
                  onChange={e => setResolutionTime(Number(e.target.value))} min={0} step={0.5}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  Agent Satisfaction <span className="text-slate-400 font-normal">({agentSatisfaction}/5)</span>
                </label>
                <StarRating value={agentSatisfaction} onChange={setAgentSatisfaction} />
              </div>
            </div>
          </div>

          <div className="border-t border-slate-100" />

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Additional Notes</label>
            <textarea value={notes} onChange={e => setNotes(e.target.value)} rows={3}
              placeholder="Any additional comments about this ticket or the prediction..."
              className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder:text-slate-400 resize-none"
            />
          </div>

          <button type="submit" disabled={loading}
            className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white text-sm font-semibold py-2.5 rounded-lg transition-colors"
          >
            {loading ? <><Loader2 size={15} className="animate-spin" /> Submitting...</> : 'Submit Feedback'}
          </button>
        </form>

        {/* Tips Card */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 self-start">
          <h3 className="text-sm font-semibold text-slate-900 mb-3">Feedback Tips</h3>
          <ul className="space-y-3 text-sm text-slate-600">
            <li className="flex gap-2">
              <span className="text-blue-500 shrink-0">•</span>
              Include the exact ticket ID for accurate tracking.
            </li>
            <li className="flex gap-2">
              <span className="text-blue-500 shrink-0">•</span>
              If the prediction was wrong, selecting the actual category helps retrain the model.
            </li>
            <li className="flex gap-2">
              <span className="text-blue-500 shrink-0">•</span>
              Resolution time helps calibrate urgency scoring.
            </li>
            <li className="flex gap-2">
              <span className="text-blue-500 shrink-0">•</span>
              Your satisfaction rating influences future model optimization.
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}
