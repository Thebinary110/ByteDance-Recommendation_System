import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import MovieRow from '../components/MovieRow'
import { SkeletonRow } from '../components/SkeletonLoader'
import { getRecommendations, getPopular, logEvent } from '../api/client'

export default function Home({ userId }) {
  const [forYou, setForYou] = useState([])
  const [popular, setPopular] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    let cancelled = false
    setLoading(true)

    Promise.all([
      getRecommendations(userId, 20).catch(() => null),
      getPopular(20).catch(() => null),
    ]).then(([recs, pop]) => {
      if (cancelled) return
      setForYou(recs?.recommendations || [])
      setPopular(Array.isArray(pop) ? pop : [])
      setLoading(false)
      logEvent(userId, 'view', null, { page: 'home' })
    }).catch(err => {
      if (!cancelled) { setError(err.message); setLoading(false) }
    })

    return () => { cancelled = true }
  }, [userId])

  // Split forYou into theme rows
  const actionMovies = forYou.filter(m => m.genres?.includes('Action'))
  const dramaMovies = forYou.filter(m => m.genres?.includes('Drama'))
  const scifiMovies = forYou.filter(m => m.genres?.includes('Sci-Fi') || m.genres?.includes('Fantasy'))

  return (
    <div className="pt-16">
      {/* Hero section */}
      <section className="relative min-h-[70vh] flex items-end bg-hero-gradient overflow-hidden">
        {/* Background orbs */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-brand-red/5 blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full bg-purple-600/5 blur-3xl" />
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-8 pb-16">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
          >
            <div className="inline-flex items-center gap-2 glass-card px-4 py-2 mb-6 text-sm text-white/60">
              AI-powered picks, just for you
            </div>
            <h1 className="text-5xl sm:text-7xl font-black text-white leading-none tracking-tight mb-4">
              Your perfect<br />
              <span className="gradient-text">movie night</span><br />
              starts here.
            </h1>
            <p className="text-white/50 text-lg max-w-xl mt-4 leading-relaxed">
              Discover films tailored to your unique taste — from hidden gems to blockbusters you can't miss.
            </p>
            <div className="flex flex-wrap gap-3 mt-8">
              <button onClick={() => navigate('/search')} className="btn-primary text-base px-8 py-3">
                Browse Films
              </button>
              <button onClick={() => navigate('/taste')} className="btn-ghost text-base px-8 py-3">
                My Taste Profile
              </button>
            </div>
          </motion.div>
        </div>

        {/* Bottom fade */}
        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[#080808] to-transparent" />
      </section>

      {/* Content rows */}
      <div className="py-4">
        {error && (
          <div className="px-8 py-4 text-amber-400/70 text-sm text-center">
            The recommendation server is offline. Start it with <code className="text-white/60">uvicorn api.main:app</code> to see personalised picks.
          </div>
        )}

        {loading ? (
          <>
            <SkeletonRow />
            <SkeletonRow />
            <SkeletonRow />
          </>
        ) : (
          <>
            <MovieRow
              title="Picked for You"
              subtitle="Based on your viewing history"
              movies={forYou}
              userId={userId}
              showExplanations
            />

            {popular.length > 0 && (
              <MovieRow
                title="Trending Now"
                subtitle="Everyone's watching these"
                movies={popular}
                userId={userId}
              />
            )}

            {actionMovies.length > 3 && (
              <MovieRow
                title="High Octane Action"
                movies={actionMovies}
                userId={userId}
              />
            )}

            {scifiMovies.length > 3 && (
              <MovieRow
                title="Worlds Beyond"
                subtitle="Sci-Fi & Fantasy picks"
                movies={scifiMovies}
                userId={userId}
              />
            )}

            {dramaMovies.length > 3 && (
              <MovieRow
                title="Emotional Journeys"
                subtitle="Drama & human stories"
                movies={dramaMovies}
                userId={userId}
              />
            )}
          </>
        )}
      </div>

      {/* Stats footer bar */}
      <motion.section
        className="mx-4 sm:mx-8 my-8 glass-card p-6"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
      >
        <div className="grid grid-cols-3 gap-6 text-center">
          {[
            { label: 'Movies catalogued', value: '27K+' },
            { label: 'Ratings analysed', value: '20M' },
            { label: 'AI model accuracy', value: 'DeepFM + MMR' },
          ].map(({ label, value }) => (
            <div key={label}>
              <div className="text-2xl font-black gradient-text">{value}</div>
              <div className="text-white/40 text-xs mt-1">{label}</div>
            </div>
          ))}
        </div>
      </motion.section>
    </div>
  )
}
