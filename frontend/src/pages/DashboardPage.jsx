import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { gestureAPI } from '../services/api';
import { motion } from 'framer-motion';
import GlassCard from '../components/GlassCard';
import { HiLightningBolt, HiEye, HiClock, HiChartBar, HiPlay, HiCog } from 'react-icons/hi';

export default function DashboardPage() {
  const { user } = useAuth();
  const [stats, setStats] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [dashRes, statusRes] = await Promise.all([
        gestureAPI.getDashboard().catch(() => null),
        gestureAPI.getStatus().catch(() => null),
      ]);
      if (dashRes) setStats(dashRes.data);
      if (statusRes) setSystemStatus(statusRes.data);
    } catch (e) {
      console.error('Dashboard load error:', e);
    } finally {
      setLoading(false);
    }
  };

  const quickActions = [
    { icon: <HiPlay size={24} />, title: 'Start Recognition', desc: 'Begin real-time sign language detection', to: '/recognition', color: 'from-neon-blue to-cyan-400' },
    { icon: <HiChartBar size={24} />, title: 'View History', desc: 'Review past recognition sessions', to: '/recognition', color: 'from-neon-purple to-violet-400' },
    { icon: <HiCog size={24} />, title: 'Settings', desc: 'Configure language and preferences', to: '/settings', color: 'from-neon-pink to-rose-400' },
  ];

  const statCards = [
    { label: 'Total Sessions', value: stats?.total_sessions || 0, icon: <HiClock size={20} />, color: 'text-neon-blue' },
    { label: 'Gestures Detected', value: stats?.total_gestures_detected || 0, icon: <HiEye size={20} />, color: 'text-neon-purple' },
    { label: 'Avg Confidence', value: `${((stats?.average_confidence || 0) * 100).toFixed(1)}%`, icon: <HiChartBar size={20} />, color: 'text-neon-green' },
    { label: 'System Status', value: systemStatus?.status === 'online' ? 'Online' : 'Demo Mode', icon: <HiLightningBolt size={20} />, color: systemStatus?.status === 'online' ? 'text-neon-green' : 'text-yellow-400' },
  ];

  const languages = { en: 'English', ta: 'Tamil', hi: 'Hindi' };

  return (
    <div className="min-h-screen bg-mesh pt-24 px-4 pb-12">
      <div className="fixed inset-0 pointer-events-none">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      <div className="max-w-6xl mx-auto relative z-10">
        {/* Welcome Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="font-display font-bold text-4xl text-white mb-2">
            Welcome back, <span className="text-gradient">{user?.full_name || 'User'}</span>
          </h1>
          <p className="text-gray-400 text-lg">
            Ready to translate sign language? Your AI assistant is {systemStatus?.status === 'online' ? 'online' : 'in demo mode'}.
          </p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {statCards.map((card, i) => (
            <GlassCard key={i} delay={i * 0.1} className="text-center">
              <div className={`${card.color} mb-2 flex justify-center`}>{card.icon}</div>
              <div className="font-display font-bold text-2xl text-white">{card.value}</div>
              <div className="text-gray-400 text-xs mt-1">{card.label}</div>
            </GlassCard>
          ))}
        </div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-8"
        >
          <h2 className="font-display font-semibold text-xl text-white mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {quickActions.map((action, i) => (
              <Link key={i} to={action.to}>
                <GlassCard delay={0.4 + i * 0.1} className="group cursor-pointer hover:neon-glow-blue transition-shadow duration-300">
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${action.color} flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform`}>
                    {action.icon}
                  </div>
                  <h3 className="font-display font-semibold text-white mb-1">{action.title}</h3>
                  <p className="text-gray-400 text-sm">{action.desc}</p>
                </GlassCard>
              </Link>
            ))}
          </div>
        </motion.div>

        {/* AI System Info & Recent Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <GlassCard hover={false} delay={0.5}>
            <h2 className="font-display font-semibold text-lg text-white mb-4 flex items-center gap-2">
              <HiLightningBolt className="text-neon-blue" /> AI System Status
            </h2>
            <div className="space-y-3">
              <div className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-gray-400 text-sm">Model</span>
                <span className="text-white text-sm font-mono">{systemStatus?.model_name || 'gesture_rf_model'}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-gray-400 text-sm">Status</span>
                <span className={`text-sm font-medium px-2 py-0.5 rounded-full ${systemStatus?.status === 'online' ? 'bg-green-500/10 text-green-400' : 'bg-yellow-500/10 text-yellow-400'}`}>
                  {systemStatus?.status === 'online' ? '● Online' : '● Demo Mode'}
                </span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-gray-400 text-sm">Supported Gestures</span>
                <span className="text-white text-sm">{systemStatus?.supported_gestures?.length || 28}</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-gray-400 text-sm">Language</span>
                <span className="text-neon-blue text-sm font-medium">{languages[user?.preferred_language] || 'English'}</span>
              </div>
            </div>
          </GlassCard>

          <GlassCard hover={false} delay={0.6}>
            <h2 className="font-display font-semibold text-lg text-white mb-4 flex items-center gap-2">
              <HiClock className="text-neon-purple" /> Recent Activity
            </h2>
            {stats?.recent_activity?.length > 0 ? (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {stats.recent_activity.slice(0, 5).map((log, i) => (
                  <div key={i} className="flex items-center justify-between py-2 px-3 rounded-lg bg-white/3 hover:bg-white/5 transition-colors">
                    <div className="flex items-center gap-3">
                      <span className="text-xl">🤟</span>
                      <div>
                        <span className="text-white text-sm font-medium">{log.gesture_detected}</span>
                        {log.translated_text && (
                          <span className="text-gray-400 text-xs ml-2">→ {log.translated_text}</span>
                        )}
                      </div>
                    </div>
                    <span className="text-xs text-gray-500">{(log.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p className="text-lg mb-2">No activity yet</p>
                <p className="text-sm">Start a recognition session to see results here</p>
              </div>
            )}
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
