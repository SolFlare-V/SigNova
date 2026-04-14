import { motion } from 'framer-motion';

export default function GlassCard({ children, className = '', hover = true, delay = 0, ...props }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      whileHover={hover ? { y: -5, transition: { duration: 0.2 } } : {}}
      className={`glass p-6 ${className}`}
      {...props}
    >
      {children}
    </motion.div>
  );
}
