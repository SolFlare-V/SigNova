import { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const res = await authAPI.getMe();
        setUser(res.data);
        setIsAuthenticated(true);
      } catch {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        setIsAuthenticated(false);
      }
    }
    setLoading(false);
  };

  const login = async (username, password) => {
    const res = await authAPI.login({ username, password });
    localStorage.setItem('token', res.data.access_token);
    const userRes = await authAPI.getMe();
    setUser(userRes.data);
    setIsAuthenticated(true);
    localStorage.setItem('user', JSON.stringify(userRes.data));
    return userRes.data;
  };

  const register = async (userData) => {
    const res = await authAPI.register(userData);
    return res.data;
  };

  const googleLogin = async (googleToken) => {
    const res = await authAPI.googleLogin(googleToken);
    localStorage.setItem('token', res.data.access_token);
    const userRes = await authAPI.getMe();
    setUser(userRes.data);
    setIsAuthenticated(true);
    localStorage.setItem('user', JSON.stringify(userRes.data));
    return userRes.data;
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setIsAuthenticated(false);
  };

  const updateUser = (updatedUser) => {
    setUser(updatedUser);
    localStorage.setItem('user', JSON.stringify(updatedUser));
  };

  return (
    <AuthContext.Provider value={{
      user, loading, isAuthenticated,
      login, register, googleLogin, logout, updateUser
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
};
