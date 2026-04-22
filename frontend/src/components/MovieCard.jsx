import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { postFeedback, logEvent } from '../api/client'

// Genre → color map for visual variety
const GENRE_COLORS = {
  Action: 'bg-red-500/20 text-red-300 border-red-500/20',
  Comedy: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/20',
  Drama: 'bg-blue-500/20 text-blue-300 border-blue-500/20',
  Horror: 'bg-purple-500/20 text-purple-300 border-purple-500/20',
  Romance: 'bg-pink-500/20 text-pink-300 border-pink-500/20',
  'Sci-Fi': 'bg-cyan-500/20 text-cyan-300 border-cyan-500/20',
  Thriller: 'bg-orange-500/20 text-orange-300 border-orange-500/20',
  Animation: 'bg-green-500/20 text-green-300 border-green-500/20',
  default: 'bg-white/5 text-white/60 border-white/10',
}

function genreClass(genre) {
  return GENRE_COLORS[genre] || GENRE_COLORS.default
}

// Deterministic color from movie_id for the poster placeholder
function posterColor(movieId) {
  const hues = [220, 280, 340, 20, 160, 60, 200, 300]
  const hue = hues[movieId % hues.length]
  return `hsl(${hue}, 40%, 18%)`
}

export default function MovieCard({ movie, userId, explanation, size = 'md', className = '' }) {
  const navigate = useNavigate()
  const [hovered, setHovered] = useState(false)
  const [feedback, setFeedback] = useState(null)

  if (!movie) return null

  const { movie_id, title, year, genres = [] } = movie

  const handleClick = () => {
    logEvent(userId, 'click', movie_id)
    navigate(`/movie/${movie_id}`)
  }

  const handleFeedback = async (action, e) => {
    e.stopPropagation()
    await postFeedback(userId, movie_id, action)
    setFeedback(action)
  }

  const isSmall = size === 'sm'
  const cardH = isSmall ? 'h-36' : 'h-48'
  const titleSize = isSmall ? 'text-sm' : 'text-base'

  return (
    <motion.div
      className={`relative rounded-xl overflow-hidden cursor-pointer flex-shrink-0 group ${className}`}
      style={{ width: isSmall ? 160 : 200 }}
      whileHover={{ scale: 1.04, y: -4 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      onClick={handleClick}
    >
      {/* Poster area */}
      <div
        className={`${cardH} w-full relative`}
        style={{ backgroundColor: posterColor(movie_id) }}
      >
        {/* Title overlay as poster text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center p-3 text-center">
          <span className="text-white/30 text-3xl font-black leading-none mb-2 select-none">
            {title.charAt(0)}
          </span>
          <span className="text-white/20 text-xs font-medium leading-snug">{year}</span>
        </div>

        {/* Gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />

        {/* Hover overlay with feedback buttons */}
        <motion.div
          className="absolute inset-0 bg-black/60 flex items-center justify-center gap-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: hovered ? 1 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <button
            onClick={e => handleFeedback('thumbs_up', e)}
            className={`w-10 h-10 rounded-full flex items-center justify-center text-lg transition-all
              ${feedback === 'thumbs_up' ? 'bg-green-500 text-white' : 'bg-white/15 hover:bg-white/25 text-white'}`}
            title="Love it"
          >
            👍
          </button>
          <button
            onClick={e => handleFeedback('thumbs_down', e)}
            className={`w-10 h-10 rounded-full flex items-center justify-center text-lg transition-all
              ${feedback === 'thumbs_down' ? 'bg-red-500 text-white' : 'bg-white/15 hover:bg-white/25 text-white'}`}
            title="Not for me"
          >
            👎
          </button>
        </motion.div>

        {/* Score badge */}
        {explanation && (
          <div className="absolute top-2 right-2">
            <div className="w-2 h-2 rounded-full bg-brand-red animate-pulse" />
          </div>
        )}
      </div>

      {/* Card info */}
      <div className="bg-[#111] p-3">
        <h3 className={`${titleSize} font-semibold text-white leading-snug line-clamp-2 mb-1.5 group-hover:text-brand-red transition-colors`}>
          {title}
        </h3>
        <div className="flex flex-wrap gap-1 mb-2">
          {genres.slice(0, 2).map(g => (
            <span
              key={g}
              className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${genreClass(g)}`}
            >
              {g}
            </span>
          ))}
        </div>
        {explanation && (
          <p className="text-[10px] text-white/40 italic leading-snug line-clamp-2">
            {explanation}
          </p>
        )}
      </div>
    </motion.div>
  )
}
