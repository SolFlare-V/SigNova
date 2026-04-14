import { useState, useRef, useCallback, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { gestureAPI, translationAPI } from '../services/api';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../components/GlassCard';
import Webcam from 'react-webcam';
import {
  HiCamera, HiStop, HiTranslate, HiLightningBolt,
  HiVolumeUp, HiRefresh, HiStatusOnline, HiStatusOffline,
} from 'react-icons/hi';

const LANGUAGES = { en: 'English', ta: 'Tamil', hi: 'Hindi' };
const CAPTURE_INTERVAL_MS = 300; // ~3 fps — fast enough for responsive recognition

export default function RecognitionPage() {
  const { user } = useAuth();
  const webcamRef   = useRef(null);
  const intervalRef = useRef(null);

  const [isRunning,          setIsRunning]          = useState(false);
  const [prediction,         setPrediction]         = useState(null);
  const [history,            setHistory]            = useState([]);
  const [sentence,           setSentence]           = useState('');
  const [translatedSentence, setTranslatedSentence] = useState('');
  const [targetLang,         setTargetLang]         = useState(user?.preferred_language || 'en');
  const [cameraReady,        setCameraReady]        = useState(false);
  const [fps,                setFps]                = useState(0);
  const [frameCount,         setFrameCount]         = useState(0);   // smoothing window fill
  const [error,              setError]              = useState(null);

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

  // ── capture one frame and call the backend ──────────────────────────────
  const captureAndPredict = useCallback(async () => {
    if (!webcamRef.current) return;
    // Request full 640×480 at high quality — MediaPipe needs clear hand edges
    const imageSrc = webcamRef.current.getScreenshot({ width: 640, height: 480 });
    if (!imageSrc) return;

    const t0 = performance.now();
    try {
      const res  = await gestureAPI.predict(imageSrc);
      const data = res.data;
      const elapsed = performance.now() - t0;
      setFps(Math.round(1000 / elapsed));
      setError(null);

      const p = data.prediction;
      setPrediction(p);
      setFrameCount(c => Math.min(c + 1, 3));   // cap at window size

      // Append to sentence only for real, confident gestures
      if (
        p.landmarks_detected &&
        p.gesture !== 'nothing' &&
        p.gesture !== 'unknown' &&
        p.gesture !== 'No gesture detected' &&
        p.confidence > 0.05
      ) {
        setHistory(prev => [p, ...prev].slice(0, 20));

        const token = p.gesture === 'space' ? ' ' : p.gesture;
        setSentence(prev => {
          if (token !== ' ' && prev.endsWith(token)) return prev;
          if (token === ' ' && prev.endsWith(' '))   return prev;
          if (token.length > 1) {
            const prefix = prev.length > 0 && !prev.endsWith(' ') ? ' ' : '';
            return prev + prefix + token + ' ';
          }
          return prev + token;
        });
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Connection error — retrying…');
    }
  }, []);

  // ── start / stop ─────────────────────────────────────────────────────────
  const startRecognition = async () => {
    try { await gestureAPI.resetBuffer(); } catch (_) {}
    setIsRunning(true);
    setSentence('');
    setTranslatedSentence('');
    setHistory([]);
    setFrameCount(0);
    setPrediction(null);
    intervalRef.current = setInterval(captureAndPredict, CAPTURE_INTERVAL_MS);
  };

  const stopRecognition = async () => {
    setIsRunning(false);
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    if (sentence && targetLang !== 'en') {
      try {
        const res = await translationAPI.translate(sentence, targetLang);
        setTranslatedSentence(res.data.translated_text);
        speakText(res.data.translated_text, targetLang);
      } catch (_) {}
    } else if (sentence) {
      speakText(sentence, 'en');
    }
  };

  const clearAll = () => {
    setSentence('');
    setTranslatedSentence('');
    setHistory([]);
    setFrameCount(0);
    setPrediction(null);
  };

  // ── TTS ──────────────────────────────────────────────────────────────────
  const speakText = (text, lang) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === 'hi' ? 'hi-IN' : lang === 'ta' ? 'ta-IN' : 'en-US';
    window.speechSynthesis.speak(u);
  };

  // ── helpers ───────────────────────────────────────────────────────────────
  const confColor = (c) => c >= 0.75 ? 'text-neon-green' : c >= 0.45 ? 'text-yellow-400' : 'text-red-400';
  const confBarColor = (c) => c >= 0.75 ? '#39ff14' : c >= 0.45 ? '#facc15' : '#f87171';
  const confPct   = (c) => `${Math.round(c * 100)}%`;

  const noHand = prediction && !prediction.landmarks_detected;
  const smoothingPct = Math.round((frameCount / 7) * 100);

  return (
    <div className="min-h-screen bg-mesh pt-24 px-4 pb-12">
      {/* background orbs */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="orb orb-1" /><div className="orb orb-2" />
      </div>

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
          <h1 className="font-display font-bold text-3xl text-white mb-2 flex items-center gap-3">
            <HiCamera className="text-neon-blue" /> Real-Time Recognition
          </h1>
          <p className="text-gray-400">Show hand gestures to the camera — AI recognises and translates in real-time.</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* ── Webcam panel ─────────────────────────────────────────────── */}
          <div className="lg:col-span-2">
            <GlassCard hover={false} className="overflow-hidden">

              {/* Camera feed */}
              <div className="relative aspect-video bg-black/60 rounded-xl overflow-hidden mb-4 border border-white/10">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  screenshotQuality={0.92}
                  className="w-full h-full object-cover"
                  videoConstraints={{ width: 640, height: 480, facingMode: 'user' }}
                  onUserMedia={() => setCameraReady(true)}
                />

                {/* Top-left badges */}
                <div className="absolute top-3 left-3 flex gap-2 flex-wrap">
                  {isRunning && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-full glass text-xs text-red-400"
                    >
                      <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" /> LIVE
                    </motion.div>
                  )}
                  {isRunning && fps > 0 && (
                    <div className="px-3 py-1 rounded-full glass text-xs text-neon-green">~{fps} FPS</div>
                  )}
                  {/* Hand status badge */}
                  {isRunning && prediction && (
                    <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full glass text-xs ${prediction.landmarks_detected ? 'text-neon-green' : 'text-gray-400'}`}>
                      {prediction.landmarks_detected
                        ? <><HiStatusOnline size={12} /> Hand detected</>
                        : <><HiStatusOffline size={12} /> No hand</>}
                    </div>
                  )}
                </div>

                {/* No-hand overlay */}
                <AnimatePresence>
                  {isRunning && noHand && (
                    <motion.div
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      className="absolute inset-0 flex items-center justify-center pointer-events-none"
                    >
                      <div className="glass px-6 py-4 rounded-2xl text-center border border-white/10">
                        <p className="text-gray-300 text-lg font-semibold">✋ No hand detected</p>
                        <p className="text-gray-500 text-sm mt-1">Show your hand to the camera</p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Prediction overlay — bottom bar */}
                <AnimatePresence>
                  {isRunning && prediction && prediction.landmarks_detected && (
                    <motion.div
                      key={prediction.gesture}
                      initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}
                      className="absolute bottom-3 left-3 right-3 glass p-3 flex items-center justify-between rounded-xl border border-white/10"
                    >
                      <div className="flex items-center gap-3">
                        <span className={`text-4xl font-display font-bold leading-none ${prediction.gesture === 'nothing' ? 'text-gray-400' : 'text-white'}`}>
                          {prediction.gesture === 'nothing' ? '—' : prediction.gesture}
                        </span>
                        {/* smoothing indicator */}
                        <div className="flex flex-col gap-0.5">
                          <span className="text-[10px] text-gray-400 uppercase tracking-wider">Smoothing</span>
                          <div className="flex gap-0.5">
                            {Array.from({ length: 3 }).map((_, i) => (
                              <div
                                key={i}
                                className={`w-2 h-2 rounded-full transition-all duration-300 ${i < frameCount ? 'bg-neon-blue' : 'bg-white/10'}`}
                              />
                            ))}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className={`text-xl font-bold ${confColor(prediction.confidence)}`}>
                          {confPct(prediction.confidence)}
                        </span>
                        <span className="text-gray-400 text-xs block">confidence</span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Error banner */}
                {error && (
                  <div className="absolute top-3 right-3 px-3 py-1 rounded-full glass text-xs text-red-400 border border-red-500/30">
                    {error}
                  </div>
                )}
              </div>

              {/* Controls row */}
              <div className="flex flex-wrap items-center gap-3">
                {!isRunning ? (
                  <button
                    onClick={startRecognition}
                    disabled={!cameraReady}
                    className="btn-neon px-7 py-2.5 rounded-xl flex items-center gap-2 disabled:opacity-40 text-sm font-semibold"
                  >
                    <HiCamera size={18} /> Start Recognition
                  </button>
                ) : (
                  <button
                    onClick={stopRecognition}
                    className="px-7 py-2.5 rounded-xl bg-red-500/20 text-red-400 border border-red-500/30 font-semibold hover:bg-red-500/30 transition-all flex items-center gap-2 text-sm"
                  >
                    <HiStop size={18} /> Stop
                  </button>
                )}

                <button
                  onClick={clearAll}
                  className="px-4 py-2.5 rounded-xl bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10 transition-all flex items-center gap-2 text-sm"
                  title="Clear sentence and history"
                >
                  <HiRefresh size={16} /> Clear
                </button>

                <select
                  value={targetLang}
                  onChange={(e) => setTargetLang(e.target.value)}
                  className="input-glass px-4 py-2.5 rounded-xl w-auto text-sm ml-auto"
                >
                  {Object.entries(LANGUAGES).map(([code, name]) => (
                    <option key={code} value={code} className="bg-dark-300">{name}</option>
                  ))}
                </select>
              </div>
            </GlassCard>
          </div>

          {/* ── Side panel ───────────────────────────────────────────────── */}
          <div className="space-y-5">

            {/* Confidence meter */}
            <GlassCard hover={false}>
              <h3 className="font-display font-semibold text-white mb-3 flex items-center gap-2 text-sm">
                <HiLightningBolt className="text-neon-blue" /> AI Confidence
              </h3>
              {/* bar */}
              <div className="h-3 rounded-full bg-white/10 overflow-hidden mb-2">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: prediction ? confBarColor(prediction.confidence) : '#ffffff20' }}
                  animate={{ width: prediction ? confPct(prediction.confidence) : '0%' }}
                  transition={{ type: 'spring', stiffness: 120, damping: 20 }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-400">
                <span>0%</span>
                <span className={prediction ? confColor(prediction.confidence) : ''}>
                  {prediction ? confPct(prediction.confidence) : '—'}
                </span>
                <span>100%</span>
              </div>

              {/* Smoothing fill indicator */}
              <div className="mt-4">
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Smoothing window</span>
                  <span className="text-neon-blue">{frameCount}/3 frames</span>
                </div>
                <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-neon-blue"
                    animate={{ width: `${smoothingPct}%` }}
                    transition={{ type: 'spring', stiffness: 120, damping: 20 }}
                  />
                </div>
              </div>
            </GlassCard>

            {/* Detected text */}
            <GlassCard hover={false}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-display font-semibold text-white flex items-center gap-2 text-sm">
                  <HiTranslate className="text-neon-purple" /> Detected Text
                </h3>
                {sentence && (
                  <button
                    onClick={() => speakText(translatedSentence || sentence, targetLang)}
                    className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-neon-blue transition-colors flex items-center gap-1 text-xs"
                  >
                    <HiVolumeUp size={14} /> Speak
                  </button>
                )}
              </div>

              <div className="min-h-[56px] p-3 rounded-xl bg-white/5 border border-white/5 mb-3">
                <p className="text-white font-mono text-lg break-words leading-snug">
                  {sentence || <span className="text-gray-600">…</span>}
                </p>
              </div>

              {translatedSentence && (
                <div className="p-3 rounded-xl bg-neon-purple/5 border border-neon-purple/20 break-words">
                  <span className="text-xs text-gray-400 block mb-1">{LANGUAGES[targetLang]}</span>
                  <p className="text-neon-purple font-medium text-sm">{translatedSentence}</p>
                </div>
              )}
            </GlassCard>

            {/* Recent detections */}
            <GlassCard hover={false}>
              <h3 className="font-display font-semibold text-white mb-3 text-sm">Recent Detections</h3>
              <div className="space-y-0.5 max-h-52 overflow-y-auto pr-1">
                {history.length > 0 ? history.map((h, i) => (
                  <div key={i} className="flex justify-between items-center py-1.5 px-2 rounded-lg hover:bg-white/5 text-sm">
                    <span className="text-white font-mono">{h.gesture}</span>
                    <div className="flex items-center gap-2">
                      {/* mini confidence pip */}
                      <div className="w-12 h-1 rounded-full bg-white/10 overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: confPct(h.confidence), backgroundColor: confBarColor(h.confidence) }}
                        />
                      </div>
                      <span className={`text-xs w-8 text-right ${confColor(h.confidence)}`}>
                        {Math.round(h.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                )) : (
                  <p className="text-gray-500 text-sm text-center py-4">No detections yet</p>
                )}
              </div>
            </GlassCard>

          </div>
        </div>
      </div>
    </div>
  );
}
