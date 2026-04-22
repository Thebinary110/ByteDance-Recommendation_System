import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { postFeedback } from '../api/client'

const MOODS = [
  { emoji: '😄', label: 'Happy', value: 'happy' },
  { emoji: '😢', label: 'Sad', value: 'sad' },
  { emoji: '😮', label: 'Surprised', value: 'surprised' },
  { emoji: '😤', label: 'Excited', value: 'excited' },
  { emoji: '😴', label: 'Bored', value: 'bored' },
  { emoji: '🤩', label: 'Amazed', value: 'amazed' },
]

export default function FeedbackModal({ isOpen, onClose, movie, userId }) {
  const [action, setAction] = useState(null)
  const [mood, setMood] = useState(null)
  const [rating, setRating] = useState(0)
  const [submitted, setSubmitted] = useState(false)

  if (!isOpen || !movie) return null

  const handleSubmit = async () => {
    if (!action) return
    await postFeedback(userId, movie.movie_id, action, {
      mood,
      rating: rating || undefined,
    })
    setSubmitted(true)
    setTimeout(() => {
      onClose()
      setAction(null)
      setMood(null)
      setRating(0)
      setSubmitted(false)
    }, 1500)
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            className="absolute inset-0 bg-black/70 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            className="relative glass-card w-full max-w-sm p-6 z-10"
            initial={{ opacity: 0, y: 60, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 60, scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          >
            {submitted ? (
              <div className="text-center py-8">
                <div className="text-5xl mb-4">✓</div>
                <p className="text-white font-semibold text-lg">Thanks for the feedback!</p>
                <p className="text-white/50 text-sm mt-1">We'll use this to improve your picks.</p>
              </div>
            ) : (
              <>
                <div className="flex items-start justify-between mb-5">
                  <div>
                    <h3 className="text-white font-bold text-lg leading-snug">
                      {movie.title}
                    </h3>
                    <p className="text-white/40 text-sm">{movie.year}</p>
                  </div>
                  <button onClick={onClose} className="text-white/40 hover:text-white ml-4 mt-1">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {/* Thumbs */}
                <div className="mb-5">
                  <p className="text-white/50 text-xs uppercase tracking-wider mb-3">How was it?</p>
                  <div className="flex gap-3">
                    {[
                      { a: 'thumbs_up', emoji: '👍', label: 'Loved it' },
                      { a: 'thumbs_down', emoji: '👎', label: 'Not for me' },
                      { a: 'skip', emoji: '⏭', label: "Haven't watched" },
                    ].map(({ a, emoji, label }) => (
                      <button
                        key={a}
                        onClick={() => setAction(a)}
                        className={`flex-1 flex flex-col items-center gap-1 py-3 rounded-xl border transition-all
                          ${action === a
                            ? 'border-brand-red bg-brand-red/20 text-white'
                            : 'border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white'}`}
                      >
                        <span className="text-xl">{emoji}</span>
                        <span className="text-xs font-medium">{label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Star rating */}
                <div className="mb-5">
                  <p className="text-white/50 text-xs uppercase tracking-wider mb-3">Star rating (optional)</p>
                  <div className="flex gap-2">
                    {[1, 2, 3, 4, 5].map(star => (
                      <button
                        key={star}
                        onClick={() => setRating(r => r === star ? 0 : star)}
                        className={`text-2xl transition-all hover:scale-110 ${
                          star <= rating ? 'text-yellow-400' : 'text-white/20'
                        }`}
                      >
                        ★
                      </button>
                    ))}
                  </div>
                </div>

                {/* Mood */}
                <div className="mb-6">
                  <p className="text-white/50 text-xs uppercase tracking-wider mb-3">Your mood while watching (optional)</p>
                  <div className="grid grid-cols-3 gap-2">
                    {MOODS.map(({ emoji, label, value }) => (
                      <button
                        key={value}
                        onClick={() => setMood(m => m === value ? null : value)}
                        className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium transition-all
                          ${mood === value
                            ? 'border-brand-red/50 bg-brand-red/15 text-white'
                            : 'border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white'}`}
                      >
                        <span>{emoji}</span>
                        <span>{label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <button
                  onClick={handleSubmit}
                  disabled={!action}
                  className={`w-full btn-primary ${!action ? 'opacity-40 cursor-not-allowed shadow-none' : ''}`}
                >
                  Submit Feedback
                </button>
              </>
            )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  )
}
