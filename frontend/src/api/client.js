import axios from 'axios'

// frontend/src/api/client.js
const BASE_URL = (import.meta.env.VITE_API_URL || "https://intimateuser6969-cinewatch-recommender.hf.space") + "/api"

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
})

// ----------------------------------------------------------------
// Recommendation endpoints
// ----------------------------------------------------------------

export const getRecommendations = (userId, limit = 20) =>
  client.get(`/recommendations/${userId}`, { params: { limit } }).then(r => r.data)

export const getMovie = (movieId) =>
  client.get(`/movies/${movieId}`).then(r => r.data)

export const getSimilarMovies = (movieId, limit = 10) =>
  client.get(`/movies/${movieId}/similar`, { params: { limit } }).then(r => r.data)

export const getPopular = (limit = 20) =>
  client.get('/movies/popular', { params: { limit } }).then(r => r.data)

export const search = (query, limit = 20) =>
  client.get('/search', { params: { q: query, limit } }).then(r => r.data)

export const getTasteProfile = (userId) =>
  client.get(`/taste-profile/${userId}`).then(r => r.data)

// ----------------------------------------------------------------
// Feedback & event logging
// ----------------------------------------------------------------

export const postFeedback = (userId, movieId, action, options = {}) =>
  client.post('/feedback', { user_id: userId, movie_id: movieId, action, ...options }).then(r => r.data)

export const logEvent = (userId, eventType, movieId = null, metadata = {}) =>
  client.post('/events', { user_id: userId, event_type: eventType, movie_id: movieId, metadata }).then(r => r.data)

export const getHealth = () =>
  client.get('/health').then(r => r.data)

export default client
