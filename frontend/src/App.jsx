import { useState } from 'react'
import TicketCategorization from './pages/TicketCategorization'
import SolutionRetrieval from './pages/SolutionRetrieval'
import Anomalies from './pages/Anomalies'
import Monitoring from './pages/Monitoring'
import Feedback from './pages/Feedback'

const PAGES = [
  { key: 'categorization', label: '📨 Ticket Categorization' },
  { key: 'solutions', label: '🔎 Solution Retrieval' },
  { key: 'anomalies', label: '🚨 Anomaly Detection' },
  { key: 'monitoring', label: '📊 Monitoring & Drift' },
  { key: 'feedback', label: '🔄 Feedback' }
]

export default function App() {
  const [page, setPage] = useState('categorization')

  return (
    <div className="app-root">
      <aside className="sidebar">
        <div className="brand">
          <h2>InsightDesk</h2>
          <small>AI Support Dashboard</small>
        </div>

        <nav className="nav">
          {PAGES.map(p => (
            <button
              key={p.key}
              className={`nav-btn ${p.key === page ? 'active' : ''}`}
              onClick={() => setPage(p.key)}
            >
              {p.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <small>API: proxied to http://localhost:8000</small>
        </div>
      </aside>

      <main className="main-content">
        {page === 'categorization' && <TicketCategorization />}
        {page === 'solutions' && <SolutionRetrieval />}
        {page === 'anomalies' && <Anomalies />}
        {page === 'monitoring' && <Monitoring />}
        {page === 'feedback' && <Feedback />}
      </main>
    </div>
  )
}
