import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { GoogleLogin } from '@react-oauth/google';
import { motion } from 'framer-motion';

export default function RegisterPage() {
  const [formData, setFormData] = useState({
    username: '', email: '', full_name: '', password: '', confirmPassword: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { register, googleLogin } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);
    setError('');
    try {
      await register({
        username: formData.username,
        email: formData.email,
        full_name: formData.full_name,
        password: formData.password,
      });
      navigate('/login');
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed. Please try again.');
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
      setError(err.response?.data?.detail || 'Google sign-up failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fields = [
    { name: 'full_name', label: 'Full Name', type: 'text', placeholder: 'Enter your full name' },
    { name: 'username', label: 'Username', type: 'text', placeholder: 'Choose a username' },
    { name: 'email', label: 'Email', type: 'email', placeholder: 'Enter your email' },
    { name: 'password', label: 'Password', type: 'password', placeholder: 'Create a password (min 6 chars)' },
    { name: 'confirmPassword', label: 'Confirm Password', type: 'password', placeholder: 'Confirm your password' },
  ];

  return (
    <div className="min-h-screen bg-mesh flex items-center justify-center px-4 py-12">
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
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring' }}
            className="w-16 h-16 rounded-2xl bg-gradient-to-br from-neon-purple to-neon-pink flex items-center justify-center mx-auto mb-4 shadow-neon-purple"
          >
            <span className="text-3xl">✨</span>
          </motion.div>
          <h1 className="font-display font-bold text-3xl text-white mb-2">Create Account</h1>
          <p className="text-gray-400">Join SignLang AI and start recognizing</p>
        </div>

        <div className="glass-strong p-8">
          {/* Google Sign-Up First */}
          <div className="flex justify-center mb-4">
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => setError('Google Sign-Up failed. Please try again.')}
              theme="filled_black"
              shape="pill"
              size="large"
              text="signup_with"
              width="100%"
            />
          </div>

          {/* Divider */}
          <div className="flex items-center gap-3 mb-5">
            <div className="flex-1 h-px bg-white/10" />
            <span className="text-gray-500 text-xs uppercase tracking-wider">or register with email</span>
            <div className="flex-1 h-px bg-white/10" />
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm"
              >
                {error}
              </motion.div>
            )}

            {fields.map((field, i) => (
              <motion.div
                key={field.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 * i }}
              >
                <label className="block text-sm font-medium text-gray-300 mb-1.5">{field.label}</label>
                <input
                  type={field.type}
                  name={field.name}
                  value={formData[field.name]}
                  onChange={handleChange}
                  className="input-glass"
                  placeholder={field.placeholder}
                  required
                  id={`register-${field.name}`}
                />
              </motion.div>
            ))}

            <button
              type="submit"
              disabled={loading}
              className="w-full btn-neon py-4 rounded-xl text-lg mt-2 disabled:opacity-50 disabled:cursor-not-allowed"
              id="register-submit"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <motion.span
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    className="inline-block w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                  />
                  Creating Account...
                </span>
              ) : (
                'Create Account'
              )}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-gray-400 text-sm">
              Already have an account?{' '}
              <Link to="/login" className="text-neon-blue hover:text-neon-purple transition-colors font-medium">
                Sign In
              </Link>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
