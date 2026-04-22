import { useState, useEffect, useRef } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { search, logEvent } from '../api/client'
import MovieCard from '../components/MovieCard'
import SkeletonLoader from '../components/SkeletonLoader'

const GENRE_FILTERS = [
  'All', 'Action', 'Comedy', 'Drama', 'Horror', 'Romance',
  'Sci-Fi', 'Thriller', 'Animation', 'Documentary',
]

export default function Search({ userId }) {
  const [params, setParams] = useSearchParams()
  const [query, setQuery] = useState(params.get('q') || '')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [activeGenre, setActiveGenre] = useState('All')
  const [hasSearched, setHasSearched] = useState(false)
  const inputRef = useRef(null)
  const navigate = useNavigate()

  // Debounced search
  useEffect(() => {
    const q = params.get('q') || ''
    if (q) { setQuery(q); doSearch(q) }
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    if (!query.trim()) { setResults([]); setHasSearched(false); return }
    const timer = setTimeout(() => doSearch(query), 350)
    return () => clearTimeout(timer)
  }, [query])

  const doSearch = async (q) => {
    if (!q.trim()) return
    setLoading(true)
    setHasSearched(true)
    try {
      const data = await search(q, 40)
      setResults(data.results || [])
      logEvent(userId, 'search', null, { query: q })
      setParams({ q })
    } catch {
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const filtered = activeGenre === 'All'
    ? results
    : results.filter(m => m.genres?.includes(activeGenre))

  return (
    <div className="pt-24 min-h-screen px-4 sm:px-8 max-w-7xl mx-auto">
      {/* Search bar */}
      <motion.div
        className="max-w-2xl mx-auto mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="relative">
          <div className="absolute left-4 top-1/2 -translate-y-1/2 text-white/40">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search any movie, genre, or year…"
            className="
              w-full bg-white/8 border border-white/15 rounded-2xl
              pl-12 pr-6 py-4 text-white text-lg placeholder-white/30
              focus:outline-none focus:border-brand-red/40 focus:bg-white/10
              transition-all duration-200
            "
          />
          {query && (
            <button
              onClick={() => { setQuery(''); setResults([]); setHasSearched(false) }}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-white/40 hover:text-white"
            >
              ✕
            </button>
          )}
        </div>
      </motion.div>

      {/* Genre filter pills */}
      <div className="flex gap-2 overflow-x-auto pb-2 mb-8 scrollbar-hide">
        {GENRE_FILTERS.map(g => (
          <button
            key={g}
            onClick={() => setActiveGenre(g)}
            className={`flex-shrink-0 px-4 py-2 rounded-full text-sm font-medium transition-all duration-200
              ${activeGenre === g
                ? 'bg-brand-red text-white shadow-lg shadow-red-900/40'
                : 'bg-white/8 border border-white/10 text-white/60 hover:text-white hover:bg-white/12'}`}
          >
            {g}
          </button>
        ))}
      </div>

      {/* Results */}
      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
          {Array.from({ length: 12 }).map((_, i) => <SkeletonLoader key={i} style={{ width: '100%' }} />)}
        </div>
      ) : hasSearched && filtered.length === 0 ? (
        <motion.div
          className="text-center py-24"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="text-6xl mb-4">🎬</div>
          <h3 className="text-white text-xl font-semibold mb-2">No results found</h3>
          <p className="text-white/40">
            {activeGenre !== 'All'
              ? `No ${activeGenre} movies match "${query}"`
              : `Nothing matched "${query}" — try different keywords`}
          </p>
        </motion.div>
      ) : !hasSearched ? (
        <motion.div
          className="text-center py-24"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="text-6xl mb-4">🔍</div>
          <h3 className="text-white text-xl font-semibold mb-2">What are you in the mood for?</h3>
          <p className="text-white/40">Search for a title, director, or genre to get started</p>
        </motion.div>
      ) : (
        <>
          <p className="text-white/40 text-sm mb-4">
            {filtered.length} result{filtered.length !== 1 ? 's' : ''} for "{query}"
            {activeGenre !== 'All' && ` in ${activeGenre}`}
          </p>
          <motion.div
            className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4"
            initial="hidden"
            animate="visible"
            variants={{
              visible: { transition: { staggerChildren: 0.04 } },
              hidden: {},
            }}
          >
            <AnimatePresence>
              {filtered.map(movie => (
                <motion.div
                  key={movie.movie_id}
                  variants={{
                    hidden: { opacity: 0, scale: 0.9 },
                    visible: { opacity: 1, scale: 1 },
                  }}
                >
                  <MovieCard
                    movie={movie}
                    userId={userId}
                    className="w-full"
                    style={{ width: '100%' }}
                  />
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        </>
      )}
    </div>
  )
}
