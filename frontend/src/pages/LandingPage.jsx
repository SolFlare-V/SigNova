import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import GlassCard from '../components/GlassCard';
import { HiLightningBolt, HiTranslate, HiEye, HiGlobe, HiShieldCheck, HiCube } from 'react-icons/hi';

const features = [
  { icon: <HiEye size={28} />, title: 'Real-Time Detection', desc: 'AI-powered gesture recognition using MediaPipe and advanced ML models with sub-300ms latency.' },
  { icon: <HiTranslate size={28} />, title: 'Multilingual Output', desc: 'Instant translation to English, Tamil, and Hindi with neural machine translation.' },
  { icon: <HiLightningBolt size={28} />, title: 'Lightning Fast', desc: 'Optimized pipeline delivering real-time predictions with confidence scoring.' },
  { icon: <HiGlobe size={28} />, title: 'Accessible Design', desc: 'Built for universal accessibility, bridging communication gaps worldwide.' },
  { icon: <HiShieldCheck size={28} />, title: 'Secure & Private', desc: 'JWT authentication, encrypted data, and privacy-first architecture.' },
  { icon: <HiCube size={28} />, title: 'Extensible Platform', desc: 'Modular design ready for CNN, Transformer, and edge AI deployment.' },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-mesh overflow-hidden">
      {/* Background Orbs */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center pt-24 px-4">
        <div className="max-w-6xl mx-auto text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass text-sm text-neon-blue mb-8"
            >
              <span className="w-2 h-2 rounded-full bg-neon-green animate-pulse" />
              AI-Powered Sign Language Recognition
            </motion.div>

            {/* Title */}
            <h1 className="font-display font-black text-5xl sm:text-6xl md:text-7xl lg:text-8xl leading-tight mb-6">
              <span className="text-white">Breaking</span>
              <br />
              <span className="text-gradient">Communication</span>
              <br />
              <span className="text-white">Barriers</span>
            </h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed"
            >
              Experience the future of accessibility with our AI-powered real-time sign language recognition system. 
              Translate gestures into text across multiple languages instantly.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <Link to="/register" className="btn-neon text-lg px-10 py-4 rounded-2xl">
                Start Recognition →
              </Link>
              <Link
                to="/login"
                className="px-10 py-4 rounded-2xl text-lg font-semibold text-gray-300 hover:text-white glass hover:bg-white/10 transition-all duration-300"
              >
                Sign In
              </Link>
            </motion.div>
          </motion.div>

          {/* Floating Hand Signs */}
          <motion.div
            animate={{ y: [0, -15, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
            className="absolute -right-10 top-1/4 text-6xl opacity-20 hidden lg:block"
          >
            🤟
          </motion.div>
          <motion.div
            animate={{ y: [0, 15, 0] }}
            transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
            className="absolute -left-10 top-1/3 text-5xl opacity-15 hidden lg:block"
          >
            ✋
          </motion.div>
          <motion.div
            animate={{ y: [0, -10, 0] }}
            transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut' }}
            className="absolute right-20 bottom-20 text-4xl opacity-15 hidden lg:block"
          >
            👋
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="font-display font-bold text-3xl md:text-5xl text-white mb-4">
              Powered by <span className="text-gradient">Advanced AI</span>
            </h2>
            <p className="text-gray-400 text-lg max-w-xl mx-auto">
              State-of-the-art technology making sign language accessible to everyone.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((f, i) => (
              <GlassCard key={i} delay={i * 0.1} className="group cursor-default">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-neon-blue/20 to-neon-purple/20 flex items-center justify-center text-neon-blue mb-4 group-hover:shadow-neon-blue transition-shadow duration-300">
                  {f.icon}
                </div>
                <h3 className="font-display font-semibold text-xl text-white mb-2">{f.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{f.desc}</p>
              </GlassCard>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="relative py-24 px-4">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="font-display font-bold text-3xl md:text-5xl text-white mb-4">
              How It <span className="text-gradient">Works</span>
            </h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { step: '01', title: 'Capture', desc: 'Camera captures your hand gestures in real-time' },
              { step: '02', title: 'Detect', desc: 'MediaPipe extracts 21 hand landmarks precisely' },
              { step: '03', title: 'Predict', desc: 'ML model classifies gesture with confidence score' },
              { step: '04', title: 'Translate', desc: 'NLP engine translates to your preferred language' },
            ].map((item, i) => (
              <GlassCard key={i} delay={i * 0.15} className="text-center relative">
                <div className="text-4xl font-display font-black text-gradient opacity-30 mb-3">{item.step}</div>
                <h3 className="font-display font-semibold text-lg text-white mb-2">{item.title}</h3>
                <p className="text-gray-400 text-sm">{item.desc}</p>
                {i < 3 && (
                  <div className="hidden md:block absolute top-1/2 -right-3 text-neon-blue/30 text-2xl">→</div>
                )}
              </GlassCard>
            ))}
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="relative py-24 px-4">
        <div className="max-w-4xl mx-auto">
          <GlassCard hover={false} className="text-center p-12 neon-glow-purple">
            <h2 className="font-display font-bold text-3xl md:text-4xl text-white mb-6">
              Our <span className="text-gradient">Mission</span>
            </h2>
            <p className="text-gray-300 text-lg leading-relaxed mb-8 max-w-2xl mx-auto">
              We believe communication is a fundamental right. Our AI-powered platform breaks down 
              barriers between deaf and hearing communities, making sign language universally 
              accessible through cutting-edge technology.
            </p>
            <Link to="/register" className="btn-neon text-lg px-10 py-4 rounded-2xl inline-block">
              Join the Movement →
            </Link>
          </GlassCard>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative py-12 px-4 border-t border-white/5">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neon-blue to-neon-purple flex items-center justify-center">
              <span className="text-sm">🤟</span>
            </div>
            <span className="font-display font-bold text-gradient">SignLang AI</span>
          </div>
          <p className="text-gray-500 text-sm">
            © 2024 SignLang AI. Built for accessibility. Powered by AI.
          </p>
          <div className="flex gap-6">
            <a href="#" className="text-gray-500 hover:text-neon-blue text-sm transition-colors">Privacy</a>
            <a href="#" className="text-gray-500 hover:text-neon-blue text-sm transition-colors">Terms</a>
            <a href="#" className="text-gray-500 hover:text-neon-blue text-sm transition-colors">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
