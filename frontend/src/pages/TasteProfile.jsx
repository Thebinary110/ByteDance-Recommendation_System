import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
} from 'recharts'
import { getTasteProfile, getRecommendations } from '../api/client'
import MovieCard from '../components/MovieCard'

const GENRE_COLORS = {
  Action: '#E50914', Comedy: '#FFD700', Drama: '#4FC3F7',
  Horror: '#AB47BC', Romance: '#F48FB1', 'Sci-Fi': '#00E5FF',
  Thriller: '#FF6D00', Animation: '#69F0AE', Documentary: '#B0BEC5',
}

function getColor(genre) {
  return GENRE_COLORS[genre] || '#6366f1'
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-card px-3 py-2 text-sm text-white">
      {payload[0]?.name}: <span className="font-bold">{(payload[0]?.value * 100).toFixed(0)}%</span>
    </div>
  )
}

export default function TasteProfile({ userId }) {
  const [profile, setProfile] = useState(null)
  const [recentRecs, setRecentRecs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      getTasteProfile(userId).catch(() => null),
      getRecommendations(userId, 6).catch(() => null),
    ]).then(([p, r]) => {
      setProfile(p)
      setRecentRecs(r?.recommendations || [])
    }).finally(() => setLoading(false))
  }, [userId])

  if (loading) {
    return (
      <div className="pt-24 min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-2 border-brand-red border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-white/40">Building your taste profile…</p>
        </div>
      </div>
    )
  }

  const genreData = profile?.genre_preferences
    ? Object.entries(profile.genre_preferences)
        .filter(([, v]) => v > 0.01)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 12)
        .map(([name, value]) => ({ name, value: Math.round(value * 100) / 100 }))
    : []

  const radarData = genreData.slice(0, 8).map(({ name, value }) => ({
    genre: name.replace('Sci-Fi', 'Sci-Fi').slice(0, 8),
    score: Math.round(value * 100),
  }))

  return (
    <div className="pt-24 min-h-screen max-w-6xl mx-auto px-4 sm:px-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-10"
      >
        <div className="flex items-center gap-4 mb-2">
          <div className="w-16 h-16 rounded-full bg-gradient-to-br from-brand-red to-brand-orange flex items-center justify-center text-2xl font-black text-white shadow-2xl shadow-red-900/40">
            {userId}
          </div>
          <div>
            <h1 className="text-3xl font-black text-white">Your Taste Profile</h1>
            <p className="text-white/40">Powered by your rating history</p>
          </div>
        </div>
      </motion.div>

      {/* Stats cards */}
      {profile && (
        <motion.div
          className="grid grid-cols-3 gap-4 mb-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          {[
            { label: 'Films Rated', value: profile.rating_count?.toLocaleString() || '—', icon: '🎬' },
            { label: 'Avg Rating', value: profile.avg_rating ? `${profile.avg_rating} ★` : '—', icon: '⭐' },
            { label: 'Top Genre', value: profile.top_genres?.[0] || '—', icon: '🏆' },
          ].map(({ label, value, icon }) => (
            <div key={label} className="glass-card p-5 text-center">
              <div className="text-2xl mb-2">{icon}</div>
              <div className="text-xl sm:text-2xl font-black text-white">{value}</div>
              <div className="text-white/40 text-xs sm:text-sm mt-1">{label}</div>
            </div>
          ))}
        </motion.div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
        {/* Bar chart — genre breakdown */}
        <motion.div
          className="glass-card p-6"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h2 className="text-white font-bold text-lg mb-5">Genre Breakdown</h2>
          {genreData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={genreData} layout="vertical" margin={{ left: 0, right: 20 }}>
                <XAxis type="number" domain={[0, 1]} hide />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                  width={80}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {genreData.map(({ name }, i) => (
                    <Cell key={i} fill={getColor(name)} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-white/30 text-sm">
              No genre data yet — start rating films!
            </div>
          )}
        </motion.div>

        {/* Radar chart */}
        <motion.div
          className="glass-card p-6"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h2 className="text-white font-bold text-lg mb-5">Taste Radar</h2>
          {radarData.length > 2 ? (
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.08)" />
                <PolarAngleAxis
                  dataKey="genre"
                  tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                />
                <Radar
                  name="Score"
                  dataKey="score"
                  stroke="#E50914"
                  fill="#E50914"
                  fillOpacity={0.25}
                />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-white/30 text-sm">
              Rate more films to unlock your radar.
            </div>
          )}
        </motion.div>
      </div>

      {/* Top genres list */}
      {profile?.top_genres?.length > 0 && (
        <motion.div
          className="glass-card p-6 mb-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <h2 className="text-white font-bold text-lg mb-4">Your Top Genres</h2>
          <div className="flex flex-wrap gap-3">
            {profile.top_genres.map((genre, i) => (
              <div
                key={genre}
                className="flex items-center gap-2 px-4 py-2 rounded-full border"
                style={{
                  borderColor: `${getColor(genre)}40`,
                  backgroundColor: `${getColor(genre)}15`,
                  color: getColor(genre),
                }}
              >
                <span className="font-black text-lg">#{i + 1}</span>
                <span className="font-semibold">{genre}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Recent picks */}
      {recentRecs.length > 0 && (
        <motion.section
          className="mb-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          <h2 className="text-2xl font-bold text-white mb-6">Your Next Watches</h2>
          <div className="flex gap-4 overflow-x-auto pb-2 scrollbar-hide">
            {recentRecs.map(movie => (
              <MovieCard key={movie.movie_id} movie={movie} userId={userId} explanation={movie.explanation} />
            ))}
          </div>
        </motion.section>
      )}
    </div>
  )
}
