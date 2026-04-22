import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { getMovie, getSimilarMovies, logEvent } from '../api/client'
import FeedbackModal from '../components/FeedbackModal'
import MovieCard from '../components/MovieCard'
import { SkeletonText } from '../components/SkeletonLoader'

function posterColor(id) {
  const hues = [220, 280, 340, 20, 160, 60, 200, 300]
  return `hsl(${hues[id % hues.length]}, 40%, 18%)`
}

export default function MovieDetail({ userId }) {
  const { id } = useParams()
  const navigate = useNavigate()
  const movieId = parseInt(id)

  const [movie, setMovie] = useState(null)
  const [similar, setSimilar] = useState([])
  const [loading, setLoading] = useState(true)
  const [feedbackOpen, setFeedbackOpen] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!movieId) return
    setLoading(true)

    Promise.all([
      getMovie(movieId),
      getSimilarMovies(movieId, 10).catch(() => ({ similar: [] })),
    ]).then(([movieData, simData]) => {
      setMovie(movieData)
      setSimilar(simData?.similar || [])
      logEvent(userId, 'view', movieId)
    }).catch(err => {
      setError(err.message)
    }).finally(() => setLoading(false))
  }, [movieId])

  if (loading) {
    return (
      <div className="pt-16 min-h-screen">
        <div className="h-72 skeleton" />
        <div className="max-w-4xl mx-auto px-6 py-8 space-y-4">
          <div className="skeleton h-10 w-2/3 rounded" />
          <SkeletonText lines={4} />
        </div>
      </div>
    )
  }

  if (error || !movie) {
    return (
      <div className="pt-24 min-h-screen flex flex-col items-center justify-center text-center px-6">
        <div className="text-6xl mb-4">🎬</div>
        <h2 className="text-white text-2xl font-bold mb-2">Movie not found</h2>
        <p className="text-white/40 mb-6">
          {error || 'This movie isn\'t in our catalog yet.'}
        </p>
        <button onClick={() => navigate(-1)} className="btn-ghost">
          ← Go back
        </button>
      </div>
    )
  }

  const { title, year, genres = [], similar_movies = [] } = movie
  const allSimilar = similar.length > 0 ? similar : similar_movies

  return (
    <div className="pt-16 min-h-screen">
      {/* Hero banner */}
      <div
        className="relative h-72 sm:h-96 w-full overflow-hidden"
        style={{ backgroundColor: posterColor(movieId) }}
      >
        {/* Decorative text */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-white/5 font-black text-[200px] leading-none select-none">
            {title.charAt(0)}
          </span>
        </div>

        {/* Gradient */}
        <div className="absolute inset-0 bg-gradient-to-t from-[#080808] via-black/30 to-transparent" />
        <div className="absolute inset-0 bg-gradient-to-r from-[#080808]/60 to-transparent" />

        {/* Back button */}
        <button
          onClick={() => navigate(-1)}
          className="absolute top-20 left-6 glass-card px-4 py-2 text-white/70 hover:text-white text-sm font-medium transition-all"
        >
          ← Back
        </button>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-4 sm:px-8 -mt-16 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main info */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              {/* Genres */}
              <div className="flex flex-wrap gap-2 mb-4">
                {genres.map(g => (
                  <span key={g} className="genre-pill">{g}</span>
                ))}
              </div>

              <h1 className="text-4xl sm:text-5xl font-black text-white leading-tight mb-2">
                {title}
              </h1>
              <p className="text-white/40 text-lg mb-8">{year}</p>

              {/* Why recommended */}
              <div className="glass-card p-5 mb-6">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-lg">🤖</span>
                  <h3 className="text-white font-semibold">Why we picked this for you</h3>
                </div>
                <p className="text-white/60 text-sm leading-relaxed">
                  This film matches your taste for{' '}
                  <span className="text-white">{genres.slice(0, 2).join(' and ')}</span>{' '}
                  films. Our AI analysed your rating history and found this aligns closely
                  with movies you've loved before. It's a top-ranked pick from users
                  with similar tastes.
                </p>
              </div>

              {/* Action buttons */}
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => { setFeedbackOpen(true); logEvent(userId, 'click', movieId) }}
                  className="btn-primary"
                >
                  Rate This Film
                </button>
                <button
                  onClick={() => logEvent(userId, 'watch', movieId, { platform: 'external' })}
                  className="btn-ghost"
                >
                  Mark as Watched
                </button>
              </div>
            </motion.div>
          </div>

          {/* Side panel */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <div className="glass-card p-5">
                <h3 className="text-white font-semibold mb-4">Details</h3>
                <dl className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <dt className="text-white/40">Release Year</dt>
                    <dd className="text-white font-medium">{year}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-white/40">Genres</dt>
                    <dd className="text-white font-medium text-right">{genres.join(', ') || '—'}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-white/40">Movie ID</dt>
                    <dd className="text-white/60 font-mono text-xs">{movieId}</dd>
                  </div>
                </dl>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Similar movies */}
        {allSimilar.length > 0 && (
          <motion.section
            className="mt-12 mb-12"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h2 className="text-2xl font-bold text-white mb-6">You Might Also Like</h2>
            <div className="flex gap-4 overflow-x-auto pb-2 scrollbar-hide">
              {allSimilar.map(m => (
                <MovieCard key={m.movie_id} movie={m} userId={userId} />
              ))}
            </div>
          </motion.section>
        )}
      </div>

      {/* Feedback modal */}
      <FeedbackModal
        isOpen={feedbackOpen}
        onClose={() => setFeedbackOpen(false)}
        movie={movie}
        userId={userId}
      />
    </div>
  )
}
