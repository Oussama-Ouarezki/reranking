import { useState } from "react";
import { Link, NavLink, Route, Routes } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import DashboardPage from "./pages/DashboardPage";
import GenerationPage from "./pages/GenerationPage";
import GenComparisonPage from "./pages/GenComparisonPage";
import StatisticalTestingPage from "./pages/StatisticalTestingPage";
import FailureAnalysisPage from "./pages/FailureAnalysisPage";
import SplashScreen from "./components/SplashScreen";

const navClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded-md text-sm font-medium transition-colors duration-150 ${
    isActive
      ? "bg-accent text-white shadow-sm"
      : "text-muted hover:text-ink hover:bg-border/40"
  }`;

export default function App() {
  const [splashDone, setSplashDone] = useState(false);

  return (
    <div className="h-full flex flex-col">
      {!splashDone && <SplashScreen onDone={() => setSplashDone(true)} />}
      <header className="h-13 px-5 border-b border-border bg-panel flex items-center justify-between shrink-0 shadow-sm">
        <Link to="/" className="flex items-center gap-2 group">
          <span className="w-6 h-6 rounded-md bg-accent flex items-center justify-center shrink-0">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="text-white">
              <circle cx="7" cy="7" r="2.5" fill="currentColor" />
              <circle cx="2" cy="4" r="1.5" fill="currentColor" opacity=".6" />
              <circle cx="12" cy="4" r="1.5" fill="currentColor" opacity=".6" />
              <circle cx="2" cy="10" r="1.5" fill="currentColor" opacity=".6" />
              <circle cx="12" cy="10" r="1.5" fill="currentColor" opacity=".6" />
              <line x1="4" y1="5" x2="5.5" y2="6.5" stroke="currentColor" strokeWidth="1" opacity=".5" />
              <line x1="10" y1="5" x2="8.5" y2="6.5" stroke="currentColor" strokeWidth="1" opacity=".5" />
              <line x1="4" y1="9" x2="5.5" y2="7.5" stroke="currentColor" strokeWidth="1" opacity=".5" />
              <line x1="10" y1="9" x2="8.5" y2="7.5" stroke="currentColor" strokeWidth="1" opacity=".5" />
            </svg>
          </span>
          <span className="font-semibold tracking-tight text-ink">
            Bio<span className="text-accent">RAG</span>
          </span>
        </Link>
        <nav className="flex gap-1">
          <NavLink to="/" end className={navClass}>Chat</NavLink>
          <NavLink to="/dashboard" className={navClass}>Retrieval</NavLink>
          <NavLink to="/generation" className={navClass}>Generation</NavLink>
          <NavLink to="/gen-comparison" className={navClass}>Gen Compare</NavLink>
          <NavLink to="/stats" className={navClass}>Stats</NavLink>
          <NavLink to="/failures" className={navClass}>Failures</NavLink>
        </nav>
      </header>
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/generation" element={<GenerationPage />} />
          <Route path="/gen-comparison" element={<GenComparisonPage />} />
          <Route path="/stats" element={<StatisticalTestingPage />} />
          <Route path="/failures" element={<FailureAnalysisPage />} />
        </Routes>
      </main>
    </div>
  );
}
