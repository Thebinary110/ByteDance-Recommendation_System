import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Search from './pages/Search'
import MovieDetail from './pages/MovieDetail'
import TasteProfile from './pages/TasteProfile'
import Feedback from './pages/Feedback'

// Demo user — in production, replace with auth context
const DEMO_USER_ID = 1

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#080808]">
        <Navbar userId={DEMO_USER_ID} />
        <main>
          <Routes>
            <Route path="/" element={<Home userId={DEMO_USER_ID} />} />
            <Route path="/search" element={<Search userId={DEMO_USER_ID} />} />
            <Route path="/movie/:id" element={<MovieDetail userId={DEMO_USER_ID} />} />
            <Route path="/taste" element={<TasteProfile userId={DEMO_USER_ID} />} />
            <Route path="/feedback" element={<Feedback userId={DEMO_USER_ID} />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
