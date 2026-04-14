import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { AnimatePresence } from 'framer-motion';
import Navbar from './components/Navbar';
import ProtectedRoute from './components/ProtectedRoute';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import RecognitionPage from './pages/RecognitionPage';
import SettingsPage from './pages/SettingsPage';

const GOOGLE_CLIENT_ID = '302779854528-cil1bag1e0gp5kemat80cqcv4sd9o1l1.apps.googleusercontent.com';

function AnimatedRoutes() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/dashboard" element={
          <ProtectedRoute><DashboardPage /></ProtectedRoute>
        } />
        <Route path="/recognition" element={
          <ProtectedRoute><RecognitionPage /></ProtectedRoute>
        } />
        <Route path="/settings" element={
          <ProtectedRoute><SettingsPage /></ProtectedRoute>
        } />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <Router>
        <AuthProvider>
          <div className="min-h-screen bg-dark-400">
            <Navbar />
            <AnimatedRoutes />
          </div>
        </AuthProvider>
      </Router>
    </GoogleOAuthProvider>
  );
}
