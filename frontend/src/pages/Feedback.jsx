import { useState } from 'react'

export default function Feedback() {
  const [ticketId, setTicketId] = useState('')
  const [predCategory, setPredCategory] = useState('')
  const [actualCategory, setActualCategory] = useState('Technical Issue')
  const [predictionCorrect, setPredictionCorrect] = useState('Yes')
  const [resolutionTime, setResolutionTime] = useState(1.0)
  const [agentSatisfaction, setAgentSatisfaction] = useState(3)
  const [notes, setNotes] = useState('')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState(null)

  async function submitAgent(e) {
    e.preventDefault(); setLoading(true); setMessage(null)
    const payload = { ticket_id: ticketId, predicted_category: predCategory, actual_category: actualCategory, prediction_correct: predictionCorrect, resolution_time: resolutionTime, agent_satisfaction: agentSatisfaction, notes, feedback_type: 'agent' }
    try {
      const res = await fetch('/api/feedback/agent', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      if (!res.ok) throw new Error(`${res.status}`)
      setMessage('Agent feedback submitted')
    } catch (e) { setMessage('Failed to submit') } finally { setLoading(false) }
  }

  return (
    <div>
      <header className="page-header"><h1>🔄 Feedback</h1><p>Agent & Customer feedback collection</p></header>

      <div className="grid">
        <form className="card form-card" onSubmit={submitAgent}>
          <label>Ticket ID</label>
          <input value={ticketId} onChange={e => setTicketId(e.target.value)} />

          <label>Predicted Category</label>
          <input value={predCategory} onChange={e => setPredCategory(e.target.value)} />

          <label>Actual Category</label>
          <select value={actualCategory} onChange={e => setActualCategory(e.target.value)}>
            <option>Technical Issue</option>
            <option>Billing</option>
            <option>Feature Request</option>
            <option>Bug Report</option>
            <option>Account</option>
            <option>Other</option>
          </select>

          <label>Prediction Correct</label>
          <select value={predictionCorrect} onChange={e => setPredictionCorrect(e.target.value)}>
            <option>Yes</option>
            <option>No</option>
            <option>Partially</option>
          </select>

          <label>Resolution Time (hours)</label>
          <input type="number" value={resolutionTime} onChange={e => setResolutionTime(Number(e.target.value))} min={0} step={0.5} />

          <label>Agent Satisfaction</label>
          <input type="range" min={1} max={5} value={agentSatisfaction} onChange={e => setAgentSatisfaction(Number(e.target.value))} />

          <label>Notes</label>
          <textarea value={notes} onChange={e => setNotes(e.target.value)} rows={4} />

          <div className="actions"><button type="submit" disabled={loading}>{loading ? 'Submitting...' : '📝 Submit Agent Feedback'}</button></div>
          {message && <div className="muted">{message}</div>}
        </form>

        <aside className="card result-card">
          <h3>Feedback Tips</h3>
          <p>Use this form to submit feedback used to improve model performance.</p>
        </aside>
      </div>
    </div>
  )
}
