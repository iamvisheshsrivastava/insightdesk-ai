import { useState } from 'react'
import { Activity, Tag, Search, AlertTriangle, BarChart2, MessageSquare } from 'lucide-react'
import TicketCategorization from './pages/TicketCategorization'
import SolutionRetrieval from './pages/SolutionRetrieval'
import Anomalies from './pages/Anomalies'
import Monitoring from './pages/Monitoring'
import Feedback from './pages/Feedback'

const PAGES = [
  { key: 'categorization', label: 'Ticket Classification', icon: Tag },
  { key: 'solutions',      label: 'Solution Retrieval',   icon: Search },
  { key: 'anomalies',      label: 'Anomaly Detection',    icon: AlertTriangle },
  { key: 'monitoring',     label: 'Monitoring & Drift',   icon: BarChart2 },
  { key: 'feedback',       label: 'Feedback',             icon: MessageSquare },
]

export default function App() {
  const [page, setPage] = useState('categorization')

  return (
    <div className="flex h-screen bg-slate-50 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-60 bg-slate-900 flex flex-col shrink-0 shadow-xl">
        {/* Brand */}
        <div className="px-5 py-5 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shrink-0">
              <Activity size={16} className="text-white" />
            </div>
            <div>
              <p className="font-bold text-white text-sm leading-tight">InsightDesk</p>
              <p className="text-slate-400 text-xs">AI Support Ops</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-0.5">
          {PAGES.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setPage(key)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 text-left ${
                page === key
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              <Icon size={15} className="shrink-0" />
              {label}
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-slate-800">
          <p className="text-slate-600 text-xs">FastAPI · React · Vite</p>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {page === 'categorization' && <TicketCategorization />}
        {page === 'solutions'      && <SolutionRetrieval />}
        {page === 'anomalies'      && <Anomalies />}
        {page === 'monitoring'     && <Monitoring />}
        {page === 'feedback'       && <Feedback />}
      </main>
    </div>
  )
}
