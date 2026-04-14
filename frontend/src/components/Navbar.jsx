import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { motion } from 'framer-motion';
import { useState } from 'react';
import { HiMenu, HiX } from 'react-icons/hi';

export default function Navbar() {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const navLinks = isAuthenticated
    ? [
        { to: '/dashboard', label: 'Dashboard' },
        { to: '/recognition', label: 'Recognition' },
        { to: '/settings', label: 'Settings' },
      ]
    : [
        { to: '/', label: 'Home' },
        { to: '/login', label: 'Login' },
        { to: '/register', label: 'Sign Up' },
      ];

  const isActive = (path) => location.pathname === path;

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="fixed top-0 left-0 right-0 z-50"
    >
      <div className="mx-4 mt-4">
        <div className="glass-strong px-6 py-3 flex items-center justify-between max-w-7xl mx-auto">
          {/* Logo */}
          <Link to={isAuthenticated ? '/dashboard' : '/'} className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-neon-blue to-neon-purple flex items-center justify-center shadow-neon-blue group-hover:shadow-neon-purple transition-shadow duration-300">
              <span className="text-white font-bold text-lg">🤟</span>
            </div>
            <span className="font-display font-bold text-xl text-gradient">SignLang AI</span>
          </Link>

          {/* Desktop Links */}
          <div className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-all duration-300 ${
                  isActive(link.to)
                    ? 'bg-white/10 text-neon-blue shadow-neon-blue'
                    : 'text-gray-300 hover:text-white hover:bg-white/5'
                }`}
              >
                {link.label}
              </Link>
            ))}
            {isAuthenticated && (
              <div className="flex items-center ml-4 gap-3">
                <span className="text-xs text-gray-400 px-3 py-1.5 rounded-full bg-white/5 border border-white/10">
                  👤 {user?.full_name || user?.username}
                </span>
                <button
                  onClick={handleLogout}
                  className="px-4 py-2 text-sm font-medium text-gray-300 hover:text-red-400 rounded-xl hover:bg-red-500/10 transition-all duration-300"
                >
                  Logout
                </button>
              </div>
            )}
          </div>

          {/* Mobile Toggle */}
          <button
            className="md:hidden text-gray-300 hover:text-white p-2"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <HiX size={24} /> : <HiMenu size={24} />}
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-strong mt-2 p-4 md:hidden max-w-7xl mx-auto"
          >
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                onClick={() => setMobileOpen(false)}
                className={`block px-4 py-3 rounded-xl text-sm font-medium mb-1 transition-all ${
                  isActive(link.to) ? 'bg-white/10 text-neon-blue' : 'text-gray-300 hover:bg-white/5'
                }`}
              >
                {link.label}
              </Link>
            ))}
            {isAuthenticated && (
              <button
                onClick={() => { handleLogout(); setMobileOpen(false); }}
                className="w-full text-left px-4 py-3 rounded-xl text-sm font-medium text-red-400 hover:bg-red-500/10 transition-all"
              >
                Logout
              </button>
            )}
          </motion.div>
        )}
      </div>
    </motion.nav>
  );
}
