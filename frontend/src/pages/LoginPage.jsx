import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { GoogleLogin } from '@react-oauth/google';
import { motion } from 'framer-motion';

export default function LoginPage() {
  const [formData, setFormData] = useState({ username: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, googleLogin } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      await login(formData.username, formData.password);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {
    setLoading(true);
    setError('');
    try {
      await googleLogin(credentialResponse.credential);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Google login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-mesh flex items-center justify-center px-4 py-12">
      {/* Background Orbs */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-md relative z-10"
      >
        {/* Header */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring' }}
            className="w-16 h-16 rounded-2xl bg-gradient-to-br from-neon-blue to-neon-purple flex items-center justify-center mx-auto mb-4 shadow-neon-blue"
          >
            <span className="text-3xl">🤟</span>
          </motion.div>
          <h1 className="font-display font-bold text-3xl text-white mb-2">Welcome Back</h1>
          <p className="text-gray-400">Sign in to continue with SignLang AI</p>
        </div>

        {/* Form */}
        <div className="glass-strong p-8">
          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm"
              >
                {error}
              </motion.div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Username</label>
              <input
                type="text"
                name="username"
                value={formData.username}
                onChange={handleChange}
                className="input-glass"
                placeholder="Enter your username"
                required
                id="login-username"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Password</label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                className="input-glass"
                placeholder="Enter your password"
                required
                id="login-password"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full btn-neon py-4 rounded-xl text-lg disabled:opacity-50 disabled:cursor-not-allowed"
              id="login-submit"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <motion.span
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    className="inline-block w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                  />
                  Signing in...
                </span>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="flex items-center gap-3 my-6">
            <div className="flex-1 h-px bg-white/10" />
            <span className="text-gray-500 text-xs uppercase tracking-wider">or continue with</span>
            <div className="flex-1 h-px bg-white/10" />
          </div>

          {/* Google Sign-In */}
          <div className="flex justify-center">
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => setError('Google Sign-In failed. Please try again.')}
              theme="filled_black"
              shape="pill"
              size="large"
              text="signin_with"
              width="100%"
            />
          </div>

          <div className="mt-6 text-center">
            <p className="text-gray-400 text-sm">
              Don't have an account?{' '}
              <Link to="/register" className="text-neon-blue hover:text-neon-purple transition-colors font-medium">
                Sign Up
              </Link>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
