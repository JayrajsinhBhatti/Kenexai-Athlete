import { useState, useEffect } from 'react';
import { api } from '../api';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend
} from 'recharts';

const COLORS = ['#6366f1', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#a78bfa'];

export default function ScoutDashboard() {
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const [tab, setTab] = useState('talents');
    const [compareIds, setCompareIds] = useState([]);
    const [comparePlayers, setComparePlayers] = useState([]);
    const [search, setSearch] = useState('');
    const [searchResults, setSearchResults] = useState([]);

    useEffect(() => {
        api.getScoutDashboard()
            .then(d => { console.log('Scout data:', d); setData(d); })
            .catch(e => { console.error('Scout error:', e); setError(e.message); });
    }, []);

    const searchPlayers = async (q) => {
        setSearch(q);
        if (q.length > 2) {
            try {
                const res = await api.getPlayers({ search: q, per_page: 10 });
                setSearchResults(res.players || []);
            } catch (e) { console.error(e); }
        } else {
            setSearchResults([]);
        }
    };

    const addToCompare = async (id) => {
        if (compareIds.length >= 2 || compareIds.includes(id)) return;
        try {
            const detail = await api.getPlayer(id);
            setCompareIds([...compareIds, id]);
            setComparePlayers([...comparePlayers, detail]);
            setSearchResults([]);
            setSearch('');
        } catch (e) { console.error(e); }
    };

    if (error) return <div className="loading-container"><p style={{ color: 'var(--danger)' }}>Error: {error}</p></div>;
    if (!data) return <div className="loading-container"><div className="spinner" /><p>Loading scout dashboard...</p></div>;

    const summary = data.summary || {};

    return (
        <div>
            <div className="page-header">
                <h2>🔍 Scout & Manager Dashboard</h2>
                <p>Discover hidden gems, compare players, and analyze recruitment opportunities</p>
            </div>

            <div className="grid-3" style={{ marginBottom: 24 }}>
                <div className="stat-card">
                    <div className="stat-icon">🌍</div>
                    <div className="stat-value">{(summary.total_scouted || 0).toLocaleString()}</div>
                    <div className="stat-label">Players Scouted</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🌟</div>
                    <div className="stat-value">{summary.young_talents_count || 0}</div>
                    <div className="stat-label">Young Talents</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📈</div>
                    <div className="stat-value">{summary.avg_potential || 0}</div>
                    <div className="stat-label">Avg Potential</div>
                </div>
            </div>

            <div className="tabs">
                {['talents', 'undervalued', 'compare', 'clusters'].map(t => (
                    <button key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
                        {t === 'talents' ? 'Young Talents' : t === 'undervalued' ? 'Undervalued' : t === 'compare' ? 'Compare Players' : 'Player Clusters'}
                    </button>
                ))}
            </div>

            {tab === 'talents' && (
                <div className="card">
                    <div className="card-header"><div className="card-title">⭐ Top Young Talents</div></div>
                    <table className="data-table">
                        <thead><tr><th>#</th><th>Player</th><th>Age</th><th>Rating</th><th>Potential</th><th>Performance</th></tr></thead>
                        <tbody>
                            {(data.young_talents || []).map((p, i) => (
                                <tr key={i}>
                                    <td style={{ fontWeight: 700, color: '#f59e0b' }}>#{i + 1}</td>
                                    <td style={{ fontWeight: 600 }}>{p.player_name}</td>
                                    <td>{typeof p.age === 'number' ? Math.round(p.age) : p.age}</td>
                                    <td><span className="badge badge-info">{p.overall_rating}</span></td>
                                    <td><span className="badge badge-success">{p.potential}</span></td>
                                    <td>{p.performance_score}</td>
                                </tr>
                            ))}
                            {(data.young_talents || []).length === 0 && <tr><td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No young talents found with current age threshold</td></tr>}
                        </tbody>
                    </table>
                </div>
            )}

            {tab === 'undervalued' && (
                <div className="card">
                    <div className="card-header"><div className="card-title">💎 Undervalued Players</div></div>
                    <table className="data-table">
                        <thead><tr><th>#</th><th>Player</th><th>Age</th><th>Rating</th><th>Performance</th><th>Value Gap</th></tr></thead>
                        <tbody>
                            {(data.undervalued_players || []).map((p, i) => (
                                <tr key={i}>
                                    <td style={{ fontWeight: 700 }}>#{i + 1}</td>
                                    <td style={{ fontWeight: 600 }}>{p.player_name}</td>
                                    <td>{typeof p.age === 'number' ? Math.round(p.age) : p.age}</td>
                                    <td>{p.overall_rating}</td>
                                    <td><span className="badge badge-success">{p.performance_score}</span></td>
                                    <td><span className="badge badge-warning">+{typeof p.value_gap === 'number' ? p.value_gap.toFixed(1) : p.value_gap}</span></td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {tab === 'compare' && (
                <div className="card">
                    <div className="card-header"><div className="card-title">⚔️ Player Comparison Tool</div></div>
                    <div className="search-bar">
                        <span>🔍</span>
                        <input placeholder="Search player to add to comparison (max 2)..." value={search} onChange={e => searchPlayers(e.target.value)} />
                    </div>
                    {searchResults.length > 0 && (
                        <div style={{ marginBottom: 16 }}>
                            {searchResults.map(p => (
                                <div key={p.player_api_id} onClick={() => addToCompare(p.player_api_id)}
                                    style={{ padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid var(--border)', fontSize: 13, display: 'flex', justifyContent: 'space-between' }}>
                                    <span>{p.player_name}</span>
                                    <span className="badge badge-info">Rating: {p.overall_rating}</span>
                                </div>
                            ))}
                        </div>
                    )}
                    {comparePlayers.length > 0 && (
                        <div>
                            <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
                                {comparePlayers.map((p, i) => (
                                    <span key={i} className={`badge ${i === 0 ? 'badge-info' : 'badge-warning'}`} style={{ padding: '6px 14px', fontSize: 13 }}>
                                        {p.player_name} ({p.overall_rating})
                                        <span onClick={() => { setComparePlayers(comparePlayers.filter((_, j) => j !== i)); setCompareIds(compareIds.filter((_, j) => j !== i)); }} style={{ marginLeft: 8, cursor: 'pointer' }}>✕</span>
                                    </span>
                                ))}
                            </div>
                            {comparePlayers.length === 2 && (
                                <div className="grid-2">
                                    <div>
                                        {['overall_rating', 'potential', 'performance_score', 'attack_score', 'midfield_score', 'defense_score', 'physical_score'].map(attr => {
                                            const v1 = comparePlayers[0][attr] || 0;
                                            const v2 = comparePlayers[1][attr] || 0;
                                            const max = Math.max(v1, v2, 1);
                                            return (
                                                <div className="comparison-row" key={attr}>
                                                    <div className="comparison-label">{attr.replace(/_/g, ' ')}</div>
                                                    <div className="comparison-bars">
                                                        <div className="comp-bar p1" style={{ width: `${(v1 / max) * 100}%` }}>{typeof v1 === 'number' ? Math.round(v1) : v1}</div>
                                                        <div className="comp-bar p2" style={{ width: `${(v2 / max) * 100}%` }}>{typeof v2 === 'number' ? Math.round(v2) : v2}</div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                    <ResponsiveContainer width="100%" height={280}>
                                        <RadarChart data={['Attack', 'Midfield', 'Defense', 'Physical', 'Overall', 'Potential'].map((s, i) => {
                                            const keys = ['attack_score', 'midfield_score', 'defense_score', 'physical_score', 'overall_rating', 'potential'];
                                            return { skill: s, p1: comparePlayers[0][keys[i]] || 0, p2: comparePlayers[1][keys[i]] || 0 };
                                        })}>
                                            <PolarGrid stroke="rgba(255,255,255,0.1)" />
                                            <PolarAngleAxis dataKey="skill" stroke="#94a3b8" fontSize={11} />
                                            <PolarRadiusAxis domain={[0, 100]} tick={false} />
                                            <Radar name={comparePlayers[0].player_name} dataKey="p1" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} />
                                            <Radar name={comparePlayers[1].player_name} dataKey="p2" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} />
                                            <Legend />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            {tab === 'clusters' && (
                <div className="grid-2">
                    {(data.player_clusters || []).map((cluster, i) => (
                        <div className="card" key={i}>
                            <div className="card-header">
                                <div>
                                    <div className="card-title" style={{ color: COLORS[i % COLORS.length] }}>{cluster.name}</div>
                                    <div className="card-subtitle">{cluster.count} players · Avg Rating: {cluster.avg_rating}</div>
                                </div>
                            </div>
                            <div style={{ display: 'flex', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
                                {Object.entries(cluster.center || {}).map(([k, v]) => (
                                    <div key={k} style={{ textAlign: 'center' }}>
                                        <div style={{ fontSize: 18, fontWeight: 700, color: COLORS[i % COLORS.length] }}>{v}</div>
                                        <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{k}</div>
                                    </div>
                                ))}
                            </div>
                            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                                <strong>Top:</strong> {(cluster.top_players || []).slice(0, 5).map(p => p.player_name).join(', ')}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
