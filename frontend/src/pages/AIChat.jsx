import { useState, useRef, useEffect, useCallback } from 'react';
import { api } from '../api';
import {
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend,
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, LineChart, Line
} from 'recharts';

const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#3b82f6'];
const HISTORY_KEY = 'athleteiq_chat_history';

function RenderChart({ chart }) {
    if (!chart || !chart.data || chart.data.length === 0) return null;
    const h = 220;
    if (chart.type === 'radar') {
        return (
            <div className="chat-chart">
                <div className="chat-chart-title">{chart.title}</div>
                <ResponsiveContainer width="100%" height={h}>
                    <RadarChart data={chart.data}>
                        <PolarGrid stroke="rgba(255,255,255,0.1)" />
                        <PolarAngleAxis dataKey="skill" stroke="#94a3b8" fontSize={10} />
                        <PolarRadiusAxis domain={[0, 100]} tick={false} />
                        <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.25} strokeWidth={2} />
                    </RadarChart>
                </ResponsiveContainer>
            </div>
        );
    }
    if (chart.type === 'radar_comparison') {
        return (
            <div className="chat-chart">
                <div className="chat-chart-title">{chart.title}</div>
                <ResponsiveContainer width="100%" height={h + 30}>
                    <RadarChart data={chart.data}>
                        <PolarGrid stroke="rgba(255,255,255,0.1)" />
                        <PolarAngleAxis dataKey="skill" stroke="#94a3b8" fontSize={10} />
                        <PolarRadiusAxis domain={[0, 100]} tick={false} />
                        <Radar name={chart.player1_name} dataKey="player1" stroke="#6366f1" fill="#6366f1" fillOpacity={0.15} />
                        <Radar name={chart.player2_name} dataKey="player2" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} />
                        <Legend />
                    </RadarChart>
                </ResponsiveContainer>
            </div>
        );
    }
    if (chart.type === 'bar') {
        const dk = Object.keys(chart.data[0]).filter(k => k !== Object.keys(chart.data[0])[0])[0] || 'value';
        return (
            <div className="chat-chart">
                <div className="chat-chart-title">{chart.title}</div>
                <ResponsiveContainer width="100%" height={h}>
                    <BarChart data={chart.data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey={Object.keys(chart.data[0])[0]} stroke="#64748b" fontSize={9} angle={-20} textAnchor="end" height={45} />
                        <YAxis stroke="#64748b" fontSize={10} />
                        <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                        <Bar dataKey={dk} fill="#6366f1" radius={[4, 4, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        );
    }
    if (chart.type === 'pie') {
        return (
            <div className="chat-chart">
                <div className="chat-chart-title">{chart.title}</div>
                <ResponsiveContainer width="100%" height={h}>
                    <PieChart>
                        <Pie data={chart.data} cx="50%" cy="50%" outerRadius={70} innerRadius={35} dataKey="value"
                            label={({ name, value }) => `${name}: ${value}`}>
                            {chart.data.map((e, i) => <Cell key={i} fill={e.color || COLORS[i % COLORS.length]} />)}
                        </Pie>
                        <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                    </PieChart>
                </ResponsiveContainer>
            </div>
        );
    }
    if (chart.type === 'line') {
        const keys = Object.keys(chart.data[0] || {}).filter(k => k !== 'date');
        return (
            <div className="chat-chart">
                <div className="chat-chart-title">{chart.title}</div>
                <ResponsiveContainer width="100%" height={h}>
                    <LineChart data={chart.data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="date" stroke="#64748b" fontSize={9} />
                        <YAxis stroke="#64748b" fontSize={10} />
                        <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                        {keys.map((key, i) => <Line key={key} type="monotone" dataKey={key} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={{ r: 2 }} name={key.replace(/_/g, ' ')} />)}
                        <Legend />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        );
    }
    return null;
}

/* ─── Markdown-like renderer ─── */
function MessageContent({ text, isStreaming }) {
    if (!text) return null;
    const renderInline = (s) => {
        const parts = [];
        let buf = '', bold = false, em = false, j = 0;
        for (let i = 0; i < s.length; i++) {
            if (s[i] === '*' && s[i + 1] === '*') {
                if (bold) { parts.push(<strong key={j++}>{buf}</strong>); buf = ''; bold = false; }
                else { if (buf) parts.push(buf); buf = ''; bold = true; }
                i++;
            } else if (s[i] === '*' && !bold) {
                if (em) { parts.push(<em key={j++}>{buf}</em>); buf = ''; em = false; }
                else { if (buf) parts.push(buf); buf = ''; em = true; }
            } else if (s[i] === '`') {
                if (buf && !bold && !em) parts.push(buf);
                buf = ''; i++;
                while (i < s.length && s[i] !== '`') { buf += s[i]; i++; }
                parts.push(<code key={j++} style={{ background: 'rgba(99,102,241,0.15)', padding: '2px 6px', borderRadius: 4, fontSize: '0.9em' }}>{buf}</code>);
                buf = '';
            } else {
                buf += s[i];
            }
        }
        if (buf) { if (bold) parts.push(<strong key={j}>{buf}</strong>); else if (em) parts.push(<em key={j}>{buf}</em>); else parts.push(buf); }
        return parts;
    };

    const lines = text.split('\n');
    const elements = [];
    let i = 0;
    while (i < lines.length) {
        const line = lines[i];
        // Table
        if (line.startsWith('|') && line.endsWith('|') && lines[i + 1]?.includes('---')) {
            const headers = line.split('|').filter(h => h.trim());
            i += 2;
            const rows = [];
            while (i < lines.length && lines[i].startsWith('|') && lines[i].endsWith('|')) {
                rows.push(lines[i].split('|').filter(c => c.trim()));
                i++;
            }
            elements.push(
                <div key={`t${i}`} className="chat-table-wrap">
                    <table className="data-table chat-table">
                        <thead><tr>{headers.map((h, j) => <th key={j}>{renderInline(h.trim())}</th>)}</tr></thead>
                        <tbody>{rows.map((row, ri) => <tr key={ri}>{row.map((cell, ci) => <td key={ci}>{renderInline(cell.trim())}</td>)}</tr>)}</tbody>
                    </table>
                </div>
            );
            continue;
        }
        // Code block
        if (line.startsWith('```')) {
            const lang = line.slice(3).trim();
            i++;
            const code = [];
            while (i < lines.length && !lines[i].startsWith('```')) { code.push(lines[i]); i++; }
            i++;
            elements.push(<pre key={`c${i}`} className="chat-code"><code>{code.join('\n')}</code></pre>);
            continue;
        }
        // Headers
        if (line.startsWith('## ')) { elements.push(<h3 key={`h${i}`} className="chat-h2">{renderInline(line.slice(3))}</h3>); i++; continue; }
        if (line.startsWith('### ')) { elements.push(<h4 key={`h${i}`} className="chat-h3">{renderInline(line.slice(4))}</h4>); i++; continue; }
        // Blockquote
        if (line.startsWith('> ')) { elements.push(<div key={`q${i}`} className="chat-quote">{renderInline(line.slice(2))}</div>); i++; continue; }
        // List
        if (line.startsWith('- ')) { elements.push(<div key={`l${i}`} className="chat-li">{renderInline(line.slice(2))}</div>); i++; continue; }
        // Empty
        if (!line.trim()) { elements.push(<div key={`e${i}`} style={{ height: 4 }} />); i++; continue; }
        // Paragraph
        elements.push(<p key={`p${i}`} className="chat-p">{renderInline(line)}</p>);
        i++;
    }

    return <div className="chat-md">{elements}{isStreaming && <span className="cursor-blink">▋</span>}</div>;
}

/* ─── MAIN COMPONENT ─── */
export default function AIChat() {
    const loadHistory = () => {
        try {
            const saved = localStorage.getItem(HISTORY_KEY);
            if (saved) return JSON.parse(saved);
        } catch { }
        return [];
    };

    const [conversations, setConversations] = useState(() => {
        const hist = loadHistory();
        return hist.length > 0 ? hist : [{
            id: Date.now(), title: 'New Chat', messages: [{
                role: 'bot', time: new Date().toISOString(),
                data: {
                    type: 'help',
                    message: "## 🤖 AthleteIQ Sports Analyst\n\nWelcome! I'm your **AI-powered sports performance analyst**. I analyze data like analysts at Manchester City and FC Barcelona.\n\n### Ask me anything:\n- 🔍 **\"Tell me about Messi\"** — Full performance analysis\n- ⚔️ **\"Compare Messi vs Ronaldo\"** — Head-to-head breakdown\n- 🏥 **\"Injury risk for Neymar\"** — Explainable risk factors\n- 📈 **\"Top 10 players\"** — Leaderboard rankings\n- 📉 **\"Show declining players\"** — Trend detection\n- ⚽ **\"Predict Barcelona vs Real Madrid\"** — Match forecast\n- 🧑‍🏫 **\"Coaching plan for Ronaldo\"** — Personalized development\n- 🏟️ **\"Lineup for Barcelona\"** — Optimal starting XI\n- 🚨 **\"Detect anomalies\"** — AI anomaly scan\n- 📊 **\"Show stats\"** — Platform overview\n",
                    suggestions: ['Top 10 players', 'Show stats overview', 'Compare Messi vs Ronaldo', 'Detect anomalies', 'Coaching plan for Neymar', 'Recommend lineup for Barcelona']
                }
            }]
        }];
    });
    const [activeConv, setActiveConv] = useState(0);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [streaming, setStreaming] = useState(false);
    const [streamText, setStreamText] = useState('');
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const messagesEnd = useRef(null);
    const inputRef = useRef(null);

    // Save to localStorage
    useEffect(() => {
        try { localStorage.setItem(HISTORY_KEY, JSON.stringify(conversations)); } catch { }
    }, [conversations]);

    useEffect(() => {
        messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
    }, [conversations, activeConv, streamText]);

    const currentMessages = conversations[activeConv]?.messages || [];

    // Simulate streaming effect
    const streamResponse = useCallback((fullText, callback) => {
        setStreaming(true);
        setStreamText('');
        let i = 0;
        const chars = fullText.split('');
        const speed = Math.max(2, Math.min(8, 2000 / chars.length)); // adaptive speed
        const interval = setInterval(() => {
            const chunk = Math.min(i + 3, chars.length); // 3 chars at a time
            setStreamText(chars.slice(0, chunk).join(''));
            i = chunk;
            if (i >= chars.length) {
                clearInterval(interval);
                setStreaming(false);
                setStreamText('');
                callback();
            }
        }, speed);
        return () => clearInterval(interval);
    }, []);

    const sendMessage = async (text) => {
        const msg = text || input.trim();
        if (!msg || loading) return;
        setInput('');

        // Add user message
        setConversations(prev => {
            const updated = [...prev];
            updated[activeConv] = {
                ...updated[activeConv],
                messages: [...updated[activeConv].messages, { role: 'user', text: msg, time: new Date().toISOString() }]
            };
            // Update title from first real message
            if (updated[activeConv].messages.filter(m => m.role === 'user').length <= 1) {
                updated[activeConv].title = msg.length > 30 ? msg.slice(0, 30) + '...' : msg;
            }
            return updated;
        });

        setLoading(true);
        try {
            const res = await api.chat(msg);
            const fullMsg = res.message || '';

            // Streaming effect
            streamResponse(fullMsg, () => {
                setConversations(prev => {
                    const updated = [...prev];
                    updated[activeConv] = {
                        ...updated[activeConv],
                        messages: [...updated[activeConv].messages, { role: 'bot', data: res, time: new Date().toISOString() }]
                    };
                    return updated;
                });
            });
        } catch (e) {
            setConversations(prev => {
                const updated = [...prev];
                updated[activeConv] = {
                    ...updated[activeConv],
                    messages: [...updated[activeConv].messages, { role: 'bot', data: { type: 'error', message: '❌ Connection error. Is the backend running?' }, time: new Date().toISOString() }]
                };
                return updated;
            });
        }
        setLoading(false);
        inputRef.current?.focus();
    };

    const newChat = () => {
        const newConv = {
            id: Date.now(), title: 'New Chat', messages: [{
                role: 'bot', time: new Date().toISOString(),
                data: { type: 'help', message: "## 🤖 New Conversation\n\nI'm ready to analyze player data, predict injuries, compare players, and more. What would you like to know?", suggestions: ['Top 10 players', 'Show stats', 'Detect anomalies', 'Show declining players'] }
            }]
        };
        setConversations(prev => [...prev, newConv]);
        setActiveConv(conversations.length);
    };

    const deleteConv = (idx) => {
        if (conversations.length <= 1) return;
        setConversations(prev => prev.filter((_, i) => i !== idx));
        setActiveConv(Math.max(0, activeConv - (idx <= activeConv ? 1 : 0)));
    };

    const clearAll = () => {
        localStorage.removeItem(HISTORY_KEY);
        const fresh = {
            id: Date.now(), title: 'New Chat', messages: [{
                role: 'bot', time: new Date().toISOString(),
                data: { type: 'help', message: "## 🤖 AthleteIQ Sports Analyst\n\nHistory cleared. Ask me anything about player performance, injuries, lineups, and more!", suggestions: ['Top 10 players', 'Show stats', 'Detect anomalies'] }
            }]
        };
        setConversations([fresh]);
        setActiveConv(0);
    };

    return (
        <div className="chat-layout">
            {/* Sidebar */}
            <div className={`chat-sidebar ${sidebarOpen ? 'open' : ''}`}>
                <div className="chat-sidebar-header">
                    <button className="btn btn-primary" style={{ width: '100%', fontSize: 13 }} onClick={newChat}>
                        ✨ New Chat
                    </button>
                </div>
                <div className="chat-history-list">
                    {conversations.map((conv, i) => (
                        <div key={conv.id} className={`chat-history-item ${i === activeConv ? 'active' : ''}`} onClick={() => setActiveConv(i)}>
                            <span className="chat-history-icon">💬</span>
                            <span className="chat-history-title">{conv.title}</span>
                            {conversations.length > 1 && (
                                <button className="chat-history-delete" onClick={e => { e.stopPropagation(); deleteConv(i); }}>🗑️</button>
                            )}
                        </div>
                    ))}
                </div>
                <div className="chat-sidebar-footer">
                    <button className="btn btn-ghost" style={{ width: '100%', fontSize: 12 }} onClick={clearAll}>🗑️ Clear All History</button>
                </div>
            </div>

            {/* Main chat area */}
            <div className="chat-main">
                <div className="chat-topbar">
                    <button className="chat-toggle-sidebar" onClick={() => setSidebarOpen(!sidebarOpen)}>
                        {sidebarOpen ? '◀' : '☰'}
                    </button>
                    <div className="chat-topbar-title">
                        <span className="chat-topbar-icon">🤖</span>
                        <div>
                            <div style={{ fontWeight: 700, fontSize: 14 }}>AthleteIQ AI Analyst</div>
                            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Professional sports analytics · Powered by ML</div>
                        </div>
                    </div>
                    <div className="chat-status">
                        <span className={`chat-status-dot ${loading ? 'thinking' : 'online'}`} />
                        {loading ? 'Thinking...' : 'Online'}
                    </div>
                </div>

                <div className="chat-messages-area">
                    {currentMessages.map((msg, i) => (
                        <div key={i} className={`chat-bubble ${msg.role}`}>
                            <div className="chat-avatar">{msg.role === 'user' ? '👤' : '🤖'}</div>
                            <div className="chat-bubble-content">
                                {msg.role === 'user' ? (
                                    <div className="chat-user-text">{msg.text}</div>
                                ) : (
                                    <div className="chat-bot-content">
                                        <MessageContent text={msg.data?.message} />
                                        {msg.data?.charts?.filter(Boolean).length > 0 && (
                                            <div className="chat-charts">{msg.data.charts.filter(Boolean).map((c, ci) => <RenderChart key={ci} chart={c} />)}</div>
                                        )}
                                        {msg.data?.suggestions?.length > 0 && (
                                            <div className="chat-suggestions">{msg.data.suggestions.map((s, si) => (
                                                <button key={si} className="chat-suggestion" onClick={() => sendMessage(s)}>{s}</button>
                                            ))}</div>
                                        )}
                                    </div>
                                )}
                                <div className="chat-time">{msg.time ? new Date(msg.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}</div>
                            </div>
                        </div>
                    ))}

                    {/* Streaming response */}
                    {streaming && (
                        <div className="chat-bubble bot">
                            <div className="chat-avatar">🤖</div>
                            <div className="chat-bubble-content">
                                <div className="chat-bot-content">
                                    <MessageContent text={streamText} isStreaming={true} />
                                </div>
                            </div>
                        </div>
                    )}

                    {loading && !streaming && (
                        <div className="chat-bubble bot">
                            <div className="chat-avatar">🤖</div>
                            <div className="chat-bubble-content">
                                <div className="chat-thinking">
                                    <span className="thinking-dot" /><span className="thinking-dot" /><span className="thinking-dot" />
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEnd} />
                </div>

                <div className="chat-input-container">
                    <div className="chat-input-box">
                        <input
                            ref={inputRef}
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                            placeholder="Ask about players, injuries, lineups, predictions..."
                            disabled={loading}
                        />
                        <button className="chat-send-btn" onClick={() => sendMessage()} disabled={loading || !input.trim()}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 2L11 13" /><path d="M22 2L15 22L11 13L2 9L22 2Z" /></svg>
                        </button>
                    </div>
                    <div className="chat-disclaimer">AthleteIQ uses ML models and synthetic data. Results are for analytical purposes only.</div>
                </div>
            </div>
        </div>
    );
}
