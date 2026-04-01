import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Leaf, Zap, BarChart3, ImageIcon, CheckCircle2,
  Loader2, AlertTriangle, Play, RefreshCw, ChevronRight,
  Database, Cpu, FlaskConical, Award, Search, ShieldAlert,
  ShieldCheck, Send, HelpCircle, X
} from 'lucide-react';
import './App.css';

// ──────────────────────────────────────────────
// Константы
// ──────────────────────────────────────────────
const WS_URL = 'ws://localhost:8000/ws/run';
const API_BASE = 'http://localhost:8000';

const STEPS = [
  { id: 1, icon: Database,    label: 'Загрузка и EDA'         },
  { id: 2, icon: Cpu,         label: 'Предобработка'          },
  { id: 3, icon: FlaskConical,label: 'Обучение моделей'       },
  { id: 4, icon: BarChart3,   label: 'Оценка и сравнение'     },
];

const METRIC_META = {
  'Accuracy':               { label: 'Accuracy',    color: '#8b5cf6', emoji: '🎯' },
  'Precision (Poisonous)':  { label: 'Precision',   color: '#06b6d4', emoji: '🔍' },
  'Recall (Poisonous)':     { label: 'Recall',      color: '#10b981', emoji: '📡' },
  'F1-Score':               { label: 'F1-Score',    color: '#f59e0b', emoji: '⚖️' },
  'ROC-AUC':                { label: 'ROC-AUC',     color: '#ec4899', emoji: '📈' },
};

// ──────────────────────────────────────────────
// Вспомогательные компоненты
// ──────────────────────────────────────────────

/** Прогресс-степпер */
function Stepper({ steps, currentStep, completedSteps }) {
  return (
    <div className="stepper">
      {steps.map((step, i) => {
        const Icon = step.icon;
        const done    = completedSteps.includes(step.id);
        const active  = currentStep === step.id;
        return (
          <React.Fragment key={step.id}>
            <div className={`step-item ${done ? 'done' : ''} ${active ? 'active' : ''}`}>
              <div className="step-icon">
                {done
                  ? <CheckCircle2 size={18} />
                  : active
                    ? <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.5, ease: 'linear' }}>
                        <Loader2 size={18} />
                      </motion.div>
                    : <Icon size={18} />
                }
              </div>
              <span className="step-label">{step.label}</span>
            </div>
            {i < steps.length - 1 && (
              <div className={`step-connector ${done ? 'done' : ''}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

/** Карточка метрики */
function MetricCard({ label, value, color, emoji, delay = 0 }) {
  const pct = Math.round(value * 100);
  return (
    <motion.div
      className="metric-card glass"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
    >
      <div className="metric-emoji">{emoji}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-value mono" style={{ color }}>{(value * 100).toFixed(4)}%</div>
      <div className="metric-bar-track">
        <motion.div
          className="metric-bar-fill"
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ delay: delay + 0.2, duration: 0.7, ease: 'easeOut' }}
        />
      </div>
    </motion.div>
  );
}

/** Секция одной модели */
function ModelSection({ result, delay }) {
  const isRF = result.Model === 'Random Forest';
  return (
    <motion.div
      className="model-section glass"
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.5 }}
    >
      <div className="model-header">
        {isRF ? <Award size={20} style={{ color: '#f59e0b' }} /> : <Zap size={20} style={{ color: '#8b5cf6' }} />}
        <h3 className="model-name">{result.Model}</h3>
        {isRF && <span className="badge-best">Лучшая</span>}
      </div>
      <div className="metrics-grid">
        {Object.entries(METRIC_META).map(([key, meta], i) => (
          <MetricCard
            key={key}
            label={meta.label}
            value={result[key]}
            color={meta.color}
            emoji={meta.emoji}
            delay={delay + i * 0.07}
          />
        ))}
      </div>
    </motion.div>
  );
}

/** Карточка графика */
function PlotCard({ src, title, delay = 0 }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <motion.div
        className="plot-card glass"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay, duration: 0.4 }}
        onClick={() => setOpen(true)}
      >
        <img src={`${API_BASE}${src}`} alt={title} className="plot-img" />
        <div className="plot-title">
          <ImageIcon size={14} style={{ opacity: 0.6 }} />
          {title}
        </div>
      </motion.div>

      {/* Lightbox */}
      <AnimatePresence>
        {open && (
          <motion.div
            className="lightbox-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setOpen(false)}
          >
            <motion.img
              src={`${API_BASE}${src}`}
              alt={title}
              className="lightbox-img"
              initial={{ scale: 0.7, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.7, opacity: 0 }}
              onClick={e => e.stopPropagation()}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

/** Лог событий */
function EventLog({ logs }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs]);
  return (
    <div className="event-log glass mono" ref={ref}>
      {logs.map((l, i) => (
        <motion.div
          key={i}
          className={`log-line ${l.type}`}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.25 }}
        >
          <span className="log-time">{l.time}</span>
          <span className="log-msg">{l.msg}</span>
        </motion.div>
      ))}
    </div>
  );
}

/** Компонент предсказания */
function PredictionPanel() {
  const [features, setFeatures] = useState(null);
  const [topFeatures, setTopFeatures] = useState([]);
  const [selectedValues, setSelectedValues] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showAll, setShowAll] = useState(false);

  // Загрузка признаков
  useEffect(() => {
    fetch(`${API_BASE}/api/features`)
      .then(r => r.json())
      .then(data => {
        setFeatures(data.all_features);
        setTopFeatures(data.top_features);
      })
      .catch(() => setError('Не удалось загрузить признаки. Убедитесь, что API запущен.'));
  }, []);

  const handleChange = (feat, val) => {
    setSelectedValues(prev => ({ ...prev, [feat]: val }));
    setResult(null);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const resp = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: selectedValues }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Ошибка предсказания');
      }
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const visibleFeatures = showAll
    ? Object.keys(features || {})
    : topFeatures;

  if (!features) {
    return (
      <div className="prediction-placeholder glass">
        {error ? (
          <div className="pred-error">
            <AlertTriangle size={20} />
            <span>{error}</span>
          </div>
        ) : (
          <div className="pred-loading">
            <Loader2 size={20} className="spin" />
            <span>Загрузка признаков…</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <motion.div
      className="prediction-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Дисклеймер */}
      <div className="disclaimer glass">
        <ShieldAlert size={18} style={{ color: '#f59e0b', flexShrink: 0 }} />
        <span>
          Бот создан исключительно в <strong>учебных целях</strong>!
          Не используйте его для реального сбора грибов.
          Всегда консультируйтесь с опытным микологом!
        </span>
      </div>

      {/* Форма */}
      <div className="feature-grid">
        {visibleFeatures.map((featKey, i) => {
          const feat = features[featKey];
          if (!feat) return null;
          return (
            <motion.div
              key={featKey}
              className="feature-select-card glass"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04, duration: 0.3 }}
            >
              <label className="feature-label">
                {feat.label}
                {topFeatures.includes(featKey) && (
                  <span className="badge-important">Важный</span>
                )}
              </label>
              <select
                className="feature-input"
                value={selectedValues[featKey] || ''}
                onChange={e => handleChange(featKey, e.target.value)}
              >
                <option value="">— Выберите —</option>
                {Object.entries(feat.options).map(([val, label]) => (
                  <option key={val} value={val}>{label}</option>
                ))}
              </select>
            </motion.div>
          );
        })}
      </div>

      {/* Кнопка «Показать все» */}
      <div className="feature-toggle-row">
        <button
          className="btn-toggle-features"
          onClick={() => setShowAll(!showAll)}
        >
          {showAll ? 'Скрыть дополнительные' : `Показать все признаки (${Object.keys(features).length})`}
        </button>
      </div>

      {/* Кнопка предсказания */}
      <motion.button
        className="btn-predict"
        disabled={loading || Object.keys(selectedValues).length === 0}
        onClick={handlePredict}
        whileHover={{ scale: 1.03 }}
        whileTap={{ scale: 0.97 }}
      >
        {loading ? (
          <><Loader2 size={18} className="spin" /> Анализируем…</>
        ) : (
          <><Send size={18} /> Получить предсказание</>
        )}
      </motion.button>

      {/* Ошибка */}
      <AnimatePresence>
        {error && (
          <motion.div
            className="pred-error"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <AlertTriangle size={16} />
            <span>{error}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Результат */}
      <AnimatePresence>
        {result && (
          <motion.div
            className={`prediction-result glass ${result.prediction.includes('Ядовитый') ? 'danger' : 'safe'}`}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
          >
            <div className="pred-result-header">
              {result.prediction.includes('Ядовитый')
                ? <ShieldAlert size={32} />
                : <ShieldCheck size={32} />
              }
              <div>
                <div className="pred-result-label">Результат анализа:</div>
                <div className="pred-result-value">{result.prediction}</div>
              </div>
            </div>
            <div className="pred-confidence">
              Уверенность модели: <strong>{(result.probability * 100).toFixed(2)}%</strong>
            </div>
            <div className="pred-disclaimer-inline">
              {result.disclaimer}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}


// ──────────────────────────────────────────────
// Главный компонент
// ──────────────────────────────────────────────
export default function App() {
  const [status, setStatus]           = useState('idle'); // idle | running | done | error
  const [currentStep, setCurrentStep] = useState(null);
  const [completedSteps, setCompletedSteps] = useState([]);
  const [results, setResults]         = useState(null);   // { results[], allPlots{} }
  const [logs, setLogs]               = useState([]);
  const [edaPlots, setEdaPlots]       = useState(null);
  const [evalPlots, setEvalPlots]     = useState(null);
  const [activeTab, setActiveTab]     = useState('dashboard'); // dashboard | predict
  const wsRef = useRef(null);

  const addLog = useCallback((msg, type = 'info') => {
    const time = new Date().toLocaleTimeString('ru-RU', { hour12: false });
    setLogs(prev => [...prev, { time, msg, type }]);
  }, []);

  const startPipeline = useCallback(() => {
    if (wsRef.current) wsRef.current.close();

    setStatus('running');
    setCurrentStep(null);
    setCompletedSteps([]);
    setResults(null);
    setEdaPlots(null);
    setEvalPlots(null);
    setLogs([]);

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => addLog('Подключение к серверу установлено…', 'success');

    ws.onmessage = ({ data }) => {
      const msg = JSON.parse(data);

      if (msg.event === 'step_start') {
        setCurrentStep(msg.step);
        const step = STEPS.find(s => s.id === msg.step);
        addLog(`▶ Шаг ${msg.step}: ${step?.label ?? msg.title}…`, 'info');
      }

      if (msg.event === 'step_done') {
        setCompletedSteps(prev => [...prev, msg.step]);
        if (msg.step === 1) setEdaPlots(msg.data.plots);
        if (msg.step === 4) setEvalPlots(msg.data.plots);
        addLog(`✓ Шаг ${msg.step} завершён`, 'success');
      }

      if (msg.event === 'pipeline_complete') {
        setStatus('done');
        setCurrentStep(null);
        setResults({ results: msg.results, allPlots: msg.all_plots });
        addLog('🎉 Пайплайн завершён успешно!', 'success');
      }

      if (msg.event === 'error') {
        setStatus('error');
        addLog(`✗ Ошибка: ${msg.message}`, 'error');
      }
    };

    ws.onerror = () => {
      setStatus('error');
      addLog('✗ Не удалось подключиться к серверу. Убедитесь, что api.py запущен.', 'error');
    };

    ws.onclose = () => {
      if (status !== 'done') addLog('Соединение закрыто.', 'muted');
    };
  }, [addLog]);

  const eda_plot_labels = {
    target_distribution:  'Распределение классов',
    features_distribution:'Признаки: odor и cap_color',
  };
  const eval_plot_labels = {
    confusion_matrix_naive_bayes:   'Матрица ошибок — Naive Bayes',
    confusion_matrix_random_forest: 'Матрица ошибок — Random Forest',
    roc_curve:                      'ROC-кривая',
  };

  return (
    <div className="app">
      {/* ── Шапка ── */}
      <header className="header glass">
        <div className="header-inner">
          <div className="logo">
            <Leaf size={28} style={{ color: '#10b981' }} />
            <div>
              <h1 className="logo-title gradient-text">Mushroom ML Dashboard</h1>
              <p className="logo-sub">Классификация грибов · Naive Bayes vs Random Forest</p>
            </div>
          </div>
          <div className="header-actions">
            {/* Табы */}
            <div className="tab-switcher">
              <button
                className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
                onClick={() => setActiveTab('dashboard')}
              >
                <BarChart3 size={16} />
                Дашборд
              </button>
              <button
                className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
                onClick={() => setActiveTab('predict')}
              >
                <Search size={16} />
                Предсказание
              </button>
            </div>

            {activeTab === 'dashboard' && (
              <motion.button
                className={`btn-run ${status === 'running' ? 'running' : ''}`}
                onClick={startPipeline}
                disabled={status === 'running'}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.97 }}
              >
                {status === 'running'
                  ? <><Loader2 size={18} className="spin" /> Выполняется…</>
                  : status === 'done'
                  ? <><RefreshCw size={18} /> Запустить снова</>
                  : <><Play size={18} /> Запустить анализ</>
                }
              </motion.button>
            )}
          </div>
        </div>
      </header>

      <main className="main">

        {/* ═══════ Вкладка: Дашборд ═══════ */}
        {activeTab === 'dashboard' && (
          <>
            {/* ── Степпер ── */}
            {(status === 'running' || status === 'done') && (
              <motion.section
                className="section"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <Stepper steps={STEPS} currentStep={currentStep} completedSteps={completedSteps} />
              </motion.section>
            )}

            {/* ── Стартовый экран ── */}
            <AnimatePresence>
              {status === 'idle' && (
                <motion.div
                  className="hero"
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <Leaf size={72} className="hero-icon" />
                  <h2 className="hero-title gradient-text">Готов к запуску</h2>
                  <p className="hero-desc">
                    Нажмите «Запустить анализ» — мы загрузим датасет,<br/>
                    обучим две модели и покажем результаты прямо здесь, с анимацией.
                  </p>
                  <motion.button
                    className="btn-run large"
                    onClick={startPipeline}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.97 }}
                  >
                    <Play size={22} /> Запустить анализ
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* ── Лог событий ── */}
            {logs.length > 0 && (
              <motion.section
                className="section"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <SectionTitle icon={ChevronRight}>Лог выполнения</SectionTitle>
                <EventLog logs={logs} />
              </motion.section>
            )}

            {/* ── EDA графики ── */}
            <AnimatePresence>
              {edaPlots && (
                <motion.section
                  className="section"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <SectionTitle icon={ImageIcon}>EDA — Исследовательский анализ</SectionTitle>
                  <div className="plots-grid">
                    {Object.entries(edaPlots).map(([key, src], i) => (
                      <PlotCard
                        key={key}
                        src={src}
                        title={eda_plot_labels[key] ?? key}
                        delay={i * 0.12}
                      />
                    ))}
                  </div>
                </motion.section>
              )}
            </AnimatePresence>

            {/* ── Метрики и eval-графики ── */}
            <AnimatePresence>
              {results && (
                <>
                  <motion.section
                    className="section"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    <SectionTitle icon={BarChart3}>Результаты моделей</SectionTitle>
                    <div className="models-grid">
                      {results.results.map((r, i) => (
                        <ModelSection key={r.Model} result={r} delay={i * 0.15} />
                      ))}
                    </div>
                  </motion.section>

                  <motion.section
                    className="section"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                  >
                    <SectionTitle icon={ImageIcon}>Матрицы ошибок и ROC-кривая</SectionTitle>
                    <div className="plots-grid three-col">
                      {evalPlots && Object.entries(evalPlots).map(([key, src], i) => (
                        <PlotCard
                          key={key}
                          src={src}
                          title={eval_plot_labels[key] ?? key}
                          delay={i * 0.1}
                        />
                      ))}
                    </div>
                  </motion.section>
                </>
              )}
            </AnimatePresence>
          </>
        )}

        {/* ═══════ Вкладка: Предсказание ═══════ */}
        {activeTab === 'predict' && (
          <motion.section
            className="section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <SectionTitle icon={Search}>Классификация гриба</SectionTitle>
            <PredictionPanel />
          </motion.section>
        )}

      </main>

      <footer className="footer">
        <span className="text-muted">Mushroom ML Dashboard · Naive Bayes vs Random Forest · Prediction API</span>
      </footer>
    </div>
  );
}

function SectionTitle({ icon: Icon, children }) {
  return (
    <div className="section-title">
      <Icon size={18} style={{ color: 'var(--primary-lt)' }} />
      <h2>{children}</h2>
    </div>
  );
}
