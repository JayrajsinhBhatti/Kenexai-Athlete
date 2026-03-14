import { useState, useRef, useEffect } from 'react';
import { api } from '../api';
import {
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend,
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, LineChart, Line
} from 'recharts';

const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#3b82f6'];

function RenderChart({ chart }) {
    if (!chart || !chart.data) return null;
    const h = 220;

    if (chart.type === 'radar') {
        return (
            <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'var(--text-muted)' }}>{chart.title}</div>
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
            <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'var(--text-muted)' }}>{chart.title}</div>
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
        return (
            <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'var(--text-muted)' }}>{chart.title}</div>
                <ResponsiveContainer width="100%" height={h}>
                    <BarChart data={chart.data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey={Object.keys(chart.data[0])[0]} stroke="#64748b" fontSize={9} angle={-30} textAnchor="end" height={50} />
                        <YAxis stroke="#64748b" fontSize={10} />
                        <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                        <Bar dataKey={Object.keys(chart.data[0])[1] || 'value'} fill="#6366f1" radius={[4, 4, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        );
    }

    if (chart.type === 'pie') {
        return (
            <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'var(--text-muted)' }}>{chart.title}</div>
                <ResponsiveContainer width="100%" height={h}>
                    <PieChart>
                        <Pie data={chart.data} cx="50%" cy="50%" outerRadius={75} innerRadius={40} dataKey="value"
                            label={({ name, value }) => `${name}: ${value}`}>
                            {chart.data.map((entry, i) => (
                                <Cell key={i} fill={entry.color || COLORS[i % COLORS.length]} />
                            ))}
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
            <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'var(--text-muted)' }}>{chart.title}</div>
                <ResponsiveContainer width="100%" height={h}>
                    <LineChart data={chart.data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="date" stroke="#64748b" fontSize={9} />
                        <YAxis stroke="#64748b" fontSize={10} />
                        <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                        {keys.map((key, i) => (
                            <Line key={key} type="monotone" dataKey={key} stroke={COLORS[i % COLORS.length]}
                                strokeWidth={2} dot={{ r: 2 }} name={key.replace(/_/g, ' ')} />
                        ))}
                        <Legend />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        );
    }

    return null;
}

function MarkdownRenderer({ text }) {
    if (!text) return null;

    const renderLine = (line, i) => {
        // Headers
        if (line.startsWith('## ')) return <h3 key={i} style={{ fontSize: 16, fontWeight: 700, margin: '14px 0 8px', borderBottom: '1px solid var(--border)', paddingBottom: 6 }}>{renderInline(line.slice(3))}</h3>;
        if (line.startsWith('### ')) return <h4 key={i} style={{ fontSize: 14, fontWeight: 600, margin: '12px 0 6px', color: 'var(--accent-primary)' }}>{renderInline(line.slice(4))}</h4>;

        // Tables
        if (line.startsWith('|') && line.endsWith('|')) return null;  // handled below

        // List items
        if (line.startsWith('- ')) return <div key={i} style={{ paddingLeft: 12, margin: '3px 0', fontSize: 13, lineHeight: 1.5 }}>{renderInline(line.slice(2))}</div>;

        // Blockquotes
        if (line.startsWith('> ')) return <div key={i} style={{ borderLeft: '3px solid var(--accent-primary)', paddingLeft: 12, margin: '8px 0', fontSize: 12, color: 'var(--text-secondary)', fontStyle: 'italic' }}>{renderInline(line.slice(2))}</div>;

        // Code blocks
        if (line.startsWith('```')) return null;

        // Empty lines
        if (!line.trim()) return <div key={i} style={{ height: 6 }} />;

        // Regular text
        return <p key={i} style={{ margin: '3px 0', fontSize: 13, lineHeight: 1.5 }}>{renderInline(line)}</p>;
    };

    const renderInline = (text) => {
        return text
            .replace(/\*\*(.*?)\*\*/g, '⟨b⟩$1⟨/b⟩')
            .replace(/\*(.*?)\*/g, '⟨i⟩$1⟨/i⟩')
            .split(/⟨(\/?\w+)⟩/)
            .reduce((acc, part, i, arr) => {
                if (part === 'b' && arr[i + 1]) {
                    acc.push(<strong key={i}>{arr[i + 1]}</strong>);
                    arr[i + 1] = '';
                } else if (part === '/b') {
                    // skip
                } else if (part === 'i' && arr[i + 1]) {
                    acc.push(<em key={i}>{arr[i + 1]}</em>);
                    arr[i + 1] = '';
                } else if (part === '/i') {
                    // skip
                } else if (part) {
                    acc.push(part);
                }
                return acc;
            }, []);
    };

    const lines = text.split('\n');

    // Extract tables
    const elements = [];
    let i = 0;
    while (i < lines.length) {
        // Detect table
        if (lines[i]?.startsWith('|') && lines[i]?.endsWith('|') && lines[i + 1]?.includes('---')) {
            const headers = lines[i].split('|').filter(h => h.trim());
            i += 2; // skip header and separator
            const rows = [];
            while (i < lines.length && lines[i].startsWith('|') && lines[i].endsWith('|')) {
                rows.push(lines[i].split('|').filter(c => c.trim()));
                i++;
            }
            elements.push(
                <div key={`table-${i}`} style={{ overflowX: 'auto', margin: '8px 0' }}>
                    <table className="data-table" style={{ fontSize: 12 }}>
                        <thead><tr>{headers.map((h, j) => <th key={j} style={{ padding: '6px 10px' }}>{renderInline(h.trim())}</th>)}</tr></thead>
                        <tbody>
                            {rows.map((row, ri) => (
                                <tr key={ri}>{row.map((cell, ci) => <td key={ci} style={{ padding: '5px 10px' }}>{renderInline(cell.trim())}</td>)}</tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            );
        } else if (lines[i]?.startsWith('```')) {
            // Code block
            i++;
            const codeLines = [];
            while (i < lines.length && !lines[i]?.startsWith('```')) {
                codeLines.push(lines[i]);
                i++;
            }
            i++; // skip closing ```
            elements.push(
                <pre key={`code-${i}`} style={{ background: 'var(--bg-glass)', padding: 12, borderRadius: 8, fontSize: 11, overflow: 'auto', margin: '6px 0' }}>
                    <code>{codeLines.join('\n')}</code>
                </pre>
            );
        } else {
            const el = renderLine(lines[i], `line-${i}`);
            if (el) elements.push(el);
            i++;
        }
    }

    return <div>{elements}</div>;
}

export default function AIChat() {
    const [messages, setMessages] = useState([
        {
            role: 'bot',
            data: {
                type: 'help',
                message: "## 🤖 AthleteIQ Sports Analyst\n\nI'm your **AI-powered sports analyst assistant**. I analyze player performance data, predict injuries, and generate professional insights like analysts at top football clubs.\n\n### What I Can Do\n- 🔍 **Player Analysis**: 'Tell me about Messi'\n- ⚔️ **Comparisons**: 'Compare Messi vs Ronaldo'\n- 🏥 **Injury Risk**: 'Injury risk for Neymar'\n- 🔋 **Fatigue Monitoring**: 'Fatigue report for Hazard'\n- 📈 **Rankings**: 'Top 10 players by attack'\n- 📉 **Trend Detection**: 'Show declining players'\n- ⚽ **Match Predictions**: 'Predict Barcelona vs Real Madrid'\n- 🧑‍🏫 **Coaching Plans**: 'Coaching plan for Ronaldo'\n- 🏟️ **Lineup AI**: 'Recommend lineup for Barcelona'\n- 🚨 **Anomaly Detection**: 'Detect anomalies'\n- 📊 **Statistics**: 'Show overall stats'\n",
                suggestions: ['Top 10 players', 'Detect anomalies', 'Compare Messi vs Ronaldo', 'Coaching plan for Neymar', 'Show stats overview']
            }
        }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEnd = useRef(null);

    useEffect(() => {
        messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const sendMessage = async (text) => {
        const msg = text || input.trim();
        if (!msg || loading) return;
        setInput('');
        setMessages(prev => [...prev, { role: 'user', text: msg }]);
        setLoading(true);
        try {
            const res = await api.chat(msg);
            setMessages(prev => [...prev, { role: 'bot', data: res }]);
        } catch {
            setMessages(prev => [...prev, { role: 'bot', data: { type: 'error', message: '❌ Something went wrong. Please try again.' } }]);
        }
        setLoading(false);
    };

    return (
        <div>
            <div className="page-header">
                <h2>🤖 AI Sports Analyst Assistant</h2>
                <p>Professional sports analytics powered by ML and generative AI</p>
            </div>

            <div className="chat-container">
                <div className="chat-messages">
                    {messages.map((msg, i) => (
                        <div key={i} className={`chat-message ${msg.role}`}>
                            {msg.role === 'user' ? (
                                <div style={{ fontSize: 14 }}>{msg.text}</div>
                            ) : (
                                <div>
                                    <MarkdownRenderer text={msg.data?.message} />

                                    {/* Render charts */}
                                    {msg.data?.charts?.filter(Boolean).length > 0 && (
                                        <div style={{ marginTop: 12 }}>
                                            {msg.data.charts.filter(Boolean).map((chart, ci) => (
                                                <RenderChart key={ci} chart={chart} />
                                            ))}
                                        </div>
                                    )}

                                    {/* Suggestions */}
                                    {msg.data?.suggestions?.length > 0 && (
                                        <div className="chat-suggestions">
                                            {msg.data.suggestions.map((s, si) => (
                                                <button key={si} className="chat-suggestion" onClick={() => sendMessage(s)}>{s}</button>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    ))}

                    {loading && (
                        <div className="chat-message bot">
                            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                                <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>Analyzing data...</span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEnd} />
                </div>

                <div className="chat-input-area">
                    <input
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && sendMessage()}
                        placeholder="Ask about players, injuries, performance, lineups, predictions..."
                        disabled={loading}
                    />
                    <button onClick={() => sendMessage()} disabled={loading || !input.trim()}>
                        Send
                    </button>
                </div>
            </div>
        </div>
    );
}
