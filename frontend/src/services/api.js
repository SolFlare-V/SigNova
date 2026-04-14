import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - attach JWT token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor - handle 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ─── Auth API ─────────────────────────────────
export const authAPI = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => api.post('/auth/login', data),
  googleLogin: (token) => api.post('/auth/google', { token }),
  getMe: () => api.get('/auth/me'),
};

// ─── User API ─────────────────────────────────
export const userAPI = {
  getProfile: () => api.get('/users/profile'),
  updateProfile: (data) => api.put('/users/profile', data),
  updateLanguage: (lang) => api.put(`/users/language/${lang}`),
  updateTheme: (theme) => api.put(`/users/theme/${theme}`),
};

// ─── Gesture API ──────────────────────────────
export const gestureAPI = {
  predict: (imageData) => api.post('/gesture/predict/public', { image_data: imageData }),
  resetBuffer: () => api.post('/gesture/reset'),
  getStatus: () => api.get('/gesture/status'),
  getDashboard: () => api.get('/gesture/dashboard'),
  getHistory: (limit = 20) => api.get(`/gesture/history?limit=${limit}`),
};

// ─── Translation API ─────────────────────────
export const translationAPI = {
  translate: (text, targetLang) => api.post('/translation/translate', { text, target_language: targetLang }),
  getLanguages: () => api.get('/translation/languages'),
};

export default api;
