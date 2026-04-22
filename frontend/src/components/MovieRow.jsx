import { useRef } from 'react'
import { motion } from 'framer-motion'
import MovieCard from './MovieCard'
import SkeletonLoader from './SkeletonLoader'

export default function MovieRow({ title, subtitle, movies, userId, loading, showExplanations = false }) {
  const scrollRef = useRef(null)

  const scroll = (dir) => {
    const el = scrollRef.current
    if (!el) return
    el.scrollBy({ left: dir * 660, behavior: 'smooth' })
  }

  return (
    <motion.section
      className="py-6"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.5 }}
    >
      {/* Row header */}
      <div className="flex items-end justify-between px-4 sm:px-8 mb-4">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold text-white">{title}</h2>
          {subtitle && <p className="text-sm text-white/40 mt-0.5">{subtitle}</p>}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => scroll(-1)}
            className="w-9 h-9 rounded-full glass-card flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 transition-all"
          >
            ‹
          </button>
          <button
            onClick={() => scroll(1)}
            className="w-9 h-9 rounded-full glass-card flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 transition-all"
          >
            ›
          </button>
        </div>
      </div>

      {/* Horizontal scroll container */}
      <div
        ref={scrollRef}
        className="flex gap-4 overflow-x-auto scrollbar-hide px-4 sm:px-8 pb-2"
        style={{ scrollSnapType: 'x mandatory' }}
      >
        {loading
          ? Array.from({ length: 8 }).map((_, i) => (
              <SkeletonLoader key={i} className="flex-shrink-0" style={{ width: 200 }} />
            ))
          : (movies || []).map((movie, i) => (
              <div key={movie.movie_id} style={{ scrollSnapAlign: 'start' }}>
                <MovieCard
                  movie={movie}
                  userId={userId}
                  explanation={showExplanations ? movie.explanation : null}
                />
              </div>
            ))
        }
      </div>
    </motion.section>
  )
}
