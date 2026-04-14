import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { userAPI } from '../services/api';
import { motion } from 'framer-motion';
import GlassCard from '../components/GlassCard';
import { HiGlobe, HiColorSwatch, HiUser, HiShieldCheck, HiSave } from 'react-icons/hi';

const LANGUAGES = { en: 'English', ta: 'தமிழ் (Tamil)', hi: 'हिन्दी (Hindi)' };

export default function SettingsPage() {
  const { user, updateUser } = useAuth();
  const [language, setLanguage] = useState(user?.preferred_language || 'en');
  const [theme, setTheme] = useState(user?.theme || 'dark');
  const [fullName, setFullName] = useState(user?.full_name || '');
  const [saving, setSaving] = useState('');
  const [success, setSuccess] = useState('');

  const saveLanguage = async () => {
    setSaving('language');
    try {
      const res = await userAPI.updateLanguage(language);
      updateUser(res.data);
      showSuccess('Language updated successfully');
    } catch (e) {
      console.error(e);
    } finally {
      setSaving('');
    }
  };

  const saveTheme = async () => {
    setSaving('theme');
    try {
      const res = await userAPI.updateTheme(theme);
      updateUser(res.data);
      showSuccess('Theme updated successfully');
    } catch (e) {
      console.error(e);
    } finally {
      setSaving('');
    }
  };

  const saveProfile = async () => {
    setSaving('profile');
    try {
      const res = await userAPI.updateProfile({ full_name: fullName });
      updateUser(res.data);
      showSuccess('Profile updated successfully');
    } catch (e) {
      console.error(e);
    } finally {
      setSaving('');
    }
  };

  const showSuccess = (msg) => {
    setSuccess(msg);
    setTimeout(() => setSuccess(''), 3000);
  };

  return (
    <div className="min-h-screen bg-mesh pt-24 px-4 pb-12">
      <div className="fixed inset-0 pointer-events-none">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
      </div>

      <div className="max-w-3xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="font-display font-bold text-3xl text-white mb-2">Settings</h1>
          <p className="text-gray-400">Manage your preferences and account settings.</p>
        </motion.div>

        {/* Success Toast */}
        {success && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mb-6 px-4 py-3 rounded-xl bg-neon-green/10 border border-neon-green/20 text-neon-green text-sm flex items-center gap-2"
          >
            ✓ {success}
          </motion.div>
        )}

        <div className="space-y-6">
          {/* Language Settings */}
          <GlassCard hover={false} delay={0.1}>
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-neon-blue to-cyan-400 flex items-center justify-center">
                <HiGlobe className="text-white" size={20} />
              </div>
              <div>
                <h2 className="font-display font-semibold text-lg text-white">Language Preference</h2>
                <p className="text-gray-400 text-sm">Choose translation target language</p>
              </div>
            </div>
            <div className="space-y-3">
              {Object.entries(LANGUAGES).map(([code, name]) => (
                <label
                  key={code}
                  className={`flex items-center justify-between p-4 rounded-xl cursor-pointer transition-all duration-200 ${
                    language === code ? 'bg-neon-blue/10 border border-neon-blue/30' : 'bg-white/3 border border-white/5 hover:bg-white/5'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <input
                      type="radio"
                      name="language"
                      value={code}
                      checked={language === code}
                      onChange={(e) => setLanguage(e.target.value)}
                      className="w-4 h-4 accent-neon-blue"
                    />
                    <span className="text-white font-medium">{name}</span>
                  </div>
                  {language === code && <span className="text-neon-blue text-xs">Selected</span>}
                </label>
              ))}
              <button
                onClick={saveLanguage}
                disabled={saving === 'language'}
                className="btn-neon px-6 py-2.5 rounded-xl text-sm flex items-center gap-2 mt-2 disabled:opacity-50"
              >
                <HiSave size={16} /> {saving === 'language' ? 'Saving...' : 'Save Language'}
              </button>
            </div>
          </GlassCard>

          {/* Theme Settings */}
          <GlassCard hover={false} delay={0.2}>
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-neon-purple to-violet-400 flex items-center justify-center">
                <HiColorSwatch className="text-white" size={20} />
              </div>
              <div>
                <h2 className="font-display font-semibold text-lg text-white">Theme</h2>
                <p className="text-gray-400 text-sm">Choose your visual preference</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'dark', label: 'Dark Mode', desc: 'Default futuristic dark theme', emoji: '🌙' },
                { value: 'midnight', label: 'Midnight', desc: 'Deep blue-black palette', emoji: '🌌' },
              ].map((t) => (
                <label
                  key={t.value}
                  className={`p-4 rounded-xl cursor-pointer transition-all duration-200 text-center ${
                    theme === t.value ? 'bg-neon-purple/10 border border-neon-purple/30' : 'bg-white/3 border border-white/5 hover:bg-white/5'
                  }`}
                >
                  <input
                    type="radio"
                    name="theme"
                    value={t.value}
                    checked={theme === t.value}
                    onChange={(e) => setTheme(e.target.value)}
                    className="hidden"
                  />
                  <span className="text-2xl block mb-2">{t.emoji}</span>
                  <span className="text-white text-sm font-medium block">{t.label}</span>
                  <span className="text-gray-400 text-xs">{t.desc}</span>
                </label>
              ))}
            </div>
            <button
              onClick={saveTheme}
              disabled={saving === 'theme'}
              className="btn-neon px-6 py-2.5 rounded-xl text-sm flex items-center gap-2 mt-4 disabled:opacity-50"
            >
              <HiSave size={16} /> {saving === 'theme' ? 'Saving...' : 'Save Theme'}
            </button>
          </GlassCard>

          {/* Account Settings */}
          <GlassCard hover={false} delay={0.3}>
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-neon-pink to-rose-400 flex items-center justify-center">
                <HiUser className="text-white" size={20} />
              </div>
              <div>
                <h2 className="font-display font-semibold text-lg text-white">Account</h2>
                <p className="text-gray-400 text-sm">Manage your profile information</p>
              </div>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Full Name</label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="input-glass"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
                <input
                  type="email"
                  value={user?.email || ''}
                  className="input-glass opacity-60 cursor-not-allowed"
                  disabled
                />
                <p className="text-xs text-gray-500 mt-1">Email cannot be changed</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Role</label>
                <div className="flex items-center gap-2">
                  <HiShieldCheck className="text-neon-blue" />
                  <span className="text-white capitalize">{user?.role || 'user'}</span>
                </div>
              </div>
              <button
                onClick={saveProfile}
                disabled={saving === 'profile'}
                className="btn-neon px-6 py-2.5 rounded-xl text-sm flex items-center gap-2 disabled:opacity-50"
              >
                <HiSave size={16} /> {saving === 'profile' ? 'Saving...' : 'Update Profile'}
              </button>
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
