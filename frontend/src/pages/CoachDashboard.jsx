import { useState, useEffect } from 'react';
import { api } from '../api';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

const COLORS = ['#6366f1', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#a78bfa'];

export default function CoachDashboard() {
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const [selectedPlayer, setSelectedPlayer] = useState(null);
    const [playerDetail, setPlayerDetail] = useState(null);

    useEffect(() => {
        api.getCoachDashboard()
            .then(d => { console.log('Coach data:', d); setData(d); })
            .catch(e => { console.error('Coach error:', e); setError(e.message); });
    }, []);

    const viewPlayer = async (id) => {
        setSelectedPlayer(id);
        try {
            const detail = await api.getPlayer(id);
            setPlayerDetail(detail);
        } catch (e) { console.error(e); }
    };

    if (error) return <div className="loading-container"><p style={{ color: 'var(--danger)' }}>Error: {error}</p></div>;
    if (!data) return <div className="loading-container"><div className="spinner" /><p>Loading coach dashboard...</p></div>;

    const summary = data.summary || {};
    const perfDistData = Object.entries(data.performance_distribution || {}).map(([k, v]) => ({ name: k, value: v }));
    const compData = Object.entries(data.squad_composition || {}).map(([k, v]) => ({ name: k, value: v }));

    const radarData = playerDetail ? [
        { skill: 'Attack', value: playerDetail.attack_score || 0 },
        { skill: 'Midfield', value: playerDetail.midfield_score || 0 },
        { skill: 'Defense', value: playerDetail.defense_score || 0 },
        { skill: 'Physical', value: playerDetail.physical_score || 0 },
        { skill: 'Overall', value: playerDetail.overall_rating || 0 },
        { skill: 'Potential', value: playerDetail.potential || 0 },
    ] : [];

    return (
        <div>
            <div className="page-header">
                <h2>🧑‍🏫 Coach Dashboard</h2>
                <p>Monitor squad performance, injury alerts, and player development</p>
            </div>

            <div className="grid-4" style={{ marginBottom: 24 }}>
                <div className="stat-card">
                    <div className="stat-icon">👥</div>
                    <div className="stat-value">{(summary.total_players || 0).toLocaleString()}</div>
                    <div className="stat-label">Total Players</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">⭐</div>
                    <div className="stat-value">{summary.avg_rating || 0}</div>
                    <div className="stat-label">Avg Rating</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📊</div>
                    <div className="stat-value">{summary.avg_performance || 0}</div>
                    <div className="stat-label">Avg Performance</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🏥</div>
                    <div className="stat-value" style={{ color: 'var(--danger)' }}>{summary.high_risk_count || 0}</div>
                    <div className="stat-label">High Injury Risk</div>
                </div>
            </div>

            {/* Player detail modal */}
            {playerDetail && (
                <div className="card" style={{ marginBottom: 20 }}>
                    <div className="card-header">
                        <div className="card-title">🔍 {playerDetail.player_name}</div>
                        <button className="btn btn-ghost" onClick={() => { setSelectedPlayer(null); setPlayerDetail(null); }}>✕ Close</button>
                    </div>
                    <div className="grid-2">
                        <div>
                            <div className="grid-3" style={{ marginBottom: 12 }}>
                                {[['overall_rating', 'Rating'], ['potential', 'Potential'], ['performance_score', 'Performance']].map(([k, l]) => (
                                    <div key={k} className="stat-card" style={{ padding: 14 }}>
                                        <div className="stat-value" style={{ fontSize: 22 }}>{playerDetail[k]}</div>
                                        <div className="stat-label">{l}</div>
                                    </div>
                                ))}
                            </div>
                            {playerDetail.injury_risk && (
                                <div style={{ padding: 12, borderRadius: 10, background: playerDetail.injury_risk.risk_level === 'High' ? 'rgba(239,68,68,0.08)' : 'rgba(16,185,129,0.08)', border: `1px solid ${playerDetail.injury_risk.risk_level === 'High' ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)'}` }}>
                                    <div style={{ fontWeight: 700, marginBottom: 4 }}>{playerDetail.injury_risk.risk_level === 'High' ? '🔴' : '🟢'} Injury Risk: {playerDetail.injury_risk.risk_level}</div>
                                    {(playerDetail.injury_risk.risk_factors || []).map((f, i) => <div key={i} style={{ fontSize: 12, color: 'var(--text-secondary)' }}>⚠️ {f}</div>)}
                                </div>
                            )}
                        </div>
                        <ResponsiveContainer width="100%" height={240}>
                            <RadarChart data={radarData}>
                                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                                <PolarAngleAxis dataKey="skill" stroke="#94a3b8" fontSize={11} />
                                <PolarRadiusAxis domain={[0, 100]} tick={false} />
                                <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} strokeWidth={2} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            <div className="grid-2" style={{ marginBottom: 24 }}>
                {/* Top performers */}
                <div className="card">
                    <div className="card-header"><div className="card-title">🏆 Top Performers</div></div>
                    <table className="data-table">
                        <thead><tr><th>#</th><th>Player</th><th>Rating</th><th>Potential</th><th>Perf</th><th></th></tr></thead>
                        <tbody>
                            {(data.top_performers || []).map((p, i) => (
                                <tr key={i}>
                                    <td style={{ fontWeight: 700, color: '#f59e0b' }}>{i + 1}</td>
                                    <td style={{ fontWeight: 600 }}>{p.player_name}</td>
                                    <td><span className="badge badge-info">{p.overall_rating}</span></td>
                                    <td>{p.potential}</td>
                                    <td><span className="badge badge-success">{p.performance_score}</span></td>
                                    <td><button className="btn btn-ghost" style={{ padding: '2px 6px', fontSize: 11 }} onClick={() => viewPlayer(p.player_api_id)}>View</button></td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Injury alerts */}
                <div className="card">
                    <div className="card-header"><div className="card-title">🏥 Injury Risk Alerts</div></div>
                    <table className="data-table">
                        <thead><tr><th>Player</th><th>Fatigue</th><th>Load</th><th>Risk</th></tr></thead>
                        <tbody>
                            {(data.injury_alerts || []).map((p, i) => (
                                <tr key={i}>
                                    <td style={{ fontWeight: 600 }}>{p.player_name}</td>
                                    <td><span className="badge badge-danger">{p.fatigue_index}</span></td>
                                    <td>{p.training_load}</td>
                                    <td><span className="badge badge-danger">{p.injury_risk}</span></td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="grid-2">
                {/* Performance distribution */}
                <div className="card">
                    <div className="card-header"><div className="card-title">📊 Performance Distribution</div></div>
                    {perfDistData.length > 0 && (
                        <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={perfDistData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                                <YAxis stroke="#64748b" fontSize={11} />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                    {perfDistData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    )}
                </div>

                {/* Squad composition */}
                <div className="card">
                    <div className="card-header"><div className="card-title">🎯 Squad Composition</div></div>
                    {compData.length > 0 && (
                        <ResponsiveContainer width="100%" height={250}>
                            <PieChart>
                                <Pie data={compData} cx="50%" cy="50%" outerRadius={90} innerRadius={45} dataKey="value"
                                    label={({ name, value }) => `${name.split(' ')[0]}: ${value}`}>
                                    {compData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                </Pie>
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                            </PieChart>
                        </ResponsiveContainer>
                    )}
                </div>
            </div>
        </div>
    );
}
