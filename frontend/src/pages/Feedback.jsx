import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { getRecommendations, postFeedback, logEvent } from '../api/client'

const MOODS = [
  { emoji: '😄', label: 'Happy', value: 'happy' },
  { emoji: '😢', label: 'Sad', value: 'sad' },
  { emoji: '😮', label: 'Surprised', value: 'surprised' },
  { emoji: '😤', label: 'Excited', value: 'excited' },
  { emoji: '😴', label: 'Bored', value: 'bored' },
  { emoji: '🤩', label: 'Amazed', value: 'amazed' },
  { emoji: '😰', label: 'Tense', value: 'tense' },
  { emoji: '🥹', label: 'Moved', value: 'moved' },
]

function posterColor(id) {
  const hues = [220, 280, 340, 20, 160, 60, 200, 300]
  return `hsl(${hues[id % hues.length]}, 40%, 18%)`
}

export default function Feedback({ userId }) {
  const [movies, setMovies] = useState([])
  const [currentIdx, setCurrentIdx] = useState(0)
  const [loading, setLoading] = useState(true)
  const [selectedMood, setSelectedMood] = useState(null)
  const [rating, setRating] = useState(0)
  const [submitted, setSubmitted] = useState(false)
  const [history, setHistory] = useState([])

  useEffect(() => {
    getRecommendations(userId, 20)
      .then(r => setMovies(r?.recommendations || []))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [userId])

  const current = movies[currentIdx]

  const handleAction = async (action) => {
    if (!current) return
    await postFeedback(userId, current.movie_id, action, {
      mood: selectedMood,
      rating: rating || undefined,
    })
    setHistory(h => [{ ...current, action, mood: selectedMood, rating }, ...h])
    setSubmitted(true)
    setTimeout(() => {
      setSubmitted(false)
      setSelectedMood(null)
      setRating(0)
      setCurrentIdx(i => i + 1)
    }, 700)
  }

  if (loading) {
    return (
      <div className="pt-24 min-h-screen flex items-center justify-center">
        <div className="w-16 h-16 border-2 border-brand-red border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (currentIdx >= movies.length || !current) {
    return (
      <div className="pt-24 min-h-screen flex flex-col items-center justify-center px-6 text-center">
        <motion.div initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}>
          <div className="text-7xl mb-6">🎉</div>
          <h2 className="text-3xl font-black text-white mb-3">All caught up!</h2>
          <p className="text-white/50 mb-8">Your feedback is improving your recommendations.</p>
          <button onClick={() => { setCurrentIdx(0); setHistory([]) }} className="btn-primary">
            Start Over
          </button>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="pt-24 min-h-screen max-w-2xl mx-auto px-4">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-black text-white mb-2">Rate & React</h1>
        <p className="text-white/40">Tell us what you think — it makes your picks better</p>
        <div className="flex items-center justify-center gap-2 mt-3">
          <div className="h-1.5 rounded-full bg-white/10 overflow-hidden" style={{ width: 200 }}>
            <div
              className="h-full bg-brand-red rounded-full transition-all duration-500"
              style={{ width: `${(currentIdx / movies.length) * 100}%` }}
            />
          </div>
          <span className="text-white/40 text-xs">{currentIdx}/{movies.length}</span>
        </div>
      </div>

      {/* Card */}
      <AnimatePresence mode="wait">
        {!submitted && (
          <motion.div
            key={current.movie_id}
            initial={{ opacity: 0, x: 60, rotate: 3 }}
            animate={{ opacity: 1, x: 0, rotate: 0 }}
            exit={{ opacity: 0, x: -80, rotate: -5 }}
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
            className="glass-card overflow-hidden mb-6"
          >
            {/* Movie banner */}
            <div
              className="h-48 relative"
              style={{ backgroundColor: posterColor(current.movie_id) }}
            >
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-white/5 font-black text-[140px] leading-none select-none">
                  {current.title?.charAt(0)}
                </span>
              </div>
              <div className="absolute inset-0 bg-gradient-to-t from-[#080808]/80 to-transparent" />
              <div className="absolute bottom-4 left-5">
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {(current.genres || []).slice(0, 3).map(g => (
                    <span key={g} className="genre-pill">{g}</span>
                  ))}
                </div>
                <h2 className="text-white font-black text-2xl leading-tight">{current.title}</h2>
                <p className="text-white/50 text-sm">{current.year}</p>
              </div>
            </div>

            {/* Body */}
            <div className="p-5">
              {/* Why recommended */}
              {current.explanation && (
                <p className="text-white/40 text-sm italic mb-5 leading-relaxed">
                  💡 {current.explanation}
                </p>
              )}

              {/* Star rating */}
              <div className="mb-5">
                <p className="text-white/50 text-xs uppercase tracking-wider mb-2">Your rating</p>
                <div className="flex gap-3">
                  {[1, 2, 3, 4, 5].map(s => (
                    <button
                      key={s}
                      onClick={() => setRating(r => r === s ? 0 : s)}
                      className={`text-3xl transition-all hover:scale-110 ${s <= rating ? 'text-yellow-400' : 'text-white/15'}`}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>

              {/* Mood */}
              <div className="mb-6">
                <p className="text-white/50 text-xs uppercase tracking-wider mb-2">How did it make you feel?</p>
                <div className="grid grid-cols-4 gap-2">
                  {MOODS.map(({ emoji, label, value }) => (
                    <button
                      key={value}
                      onClick={() => setSelectedMood(m => m === value ? null : value)}
                      className={`flex flex-col items-center gap-1 py-2.5 rounded-xl border text-xs font-medium transition-all
                        ${selectedMood === value
                          ? 'border-brand-red/50 bg-brand-red/15 text-white'
                          : 'border-white/8 bg-white/4 text-white/50 hover:text-white hover:bg-white/8'}`}
                    >
                      <span className="text-xl">{emoji}</span>
                      <span>{label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Action buttons */}
              <div className="grid grid-cols-3 gap-3">
                <button
                  onClick={() => handleAction('thumbs_up')}
                  className="flex flex-col items-center gap-2 py-4 rounded-xl bg-green-500/15 border border-green-500/20 text-green-400 hover:bg-green-500/25 transition-all hover:scale-105"
                >
                  <span className="text-2xl">👍</span>
                  <span className="text-xs font-semibold">Love it</span>
                </button>
                <button
                  onClick={() => handleAction('skip')}
                  className="flex flex-col items-center gap-2 py-4 rounded-xl bg-white/5 border border-white/8 text-white/50 hover:bg-white/10 hover:text-white transition-all hover:scale-105"
                >
                  <span className="text-2xl">⏭</span>
                  <span className="text-xs font-semibold">Skip</span>
                </button>
                <button
                  onClick={() => handleAction('thumbs_down')}
                  className="flex flex-col items-center gap-2 py-4 rounded-xl bg-brand-red/10 border border-brand-red/20 text-red-400 hover:bg-brand-red/20 transition-all hover:scale-105"
                >
                  <span className="text-2xl">👎</span>
                  <span className="text-xs font-semibold">Not for me</span>
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {submitted && (
          <motion.div
            key="submitted"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center justify-center h-64"
          >
            <div className="text-center">
              <div className="text-5xl mb-3">✓</div>
              <p className="text-white font-semibold">Saved!</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Recent feedback history */}
      {history.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-12"
        >
          <h3 className="text-white/50 text-sm font-medium uppercase tracking-wider mb-3">
            Recently rated
          </h3>
          <div className="space-y-2">
            {history.slice(0, 5).map((item, i) => (
              <div key={i} className="flex items-center gap-3 glass-card px-4 py-3 text-sm">
                <span className="text-xl">
                  {item.action === 'thumbs_up' ? '👍' : item.action === 'thumbs_down' ? '👎' : '⏭'}
                </span>
                <span className="text-white flex-1 truncate">{item.title}</span>
                {item.rating > 0 && (
                  <span className="text-yellow-400 font-bold">{item.rating}★</span>
                )}
                {item.mood && <span className="text-white/30">{item.mood}</span>}
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}
