const BASE = 'http://localhost:5000';

async function fetchJSON(url) {
  const res = await fetch(`${BASE}${url}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  // Data Quality & EDA
  getDataQuality: () => fetchJSON('/api/data-quality'),
  getEDA: () => fetchJSON('/api/eda'),

  // Players
  getPlayers: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return fetchJSON(`/api/players?${q}`);
  },
  getPlayer: (id) => fetchJSON(`/api/players/${id}`),
  getPlayerInjuryRisk: (id) => fetchJSON(`/api/players/${id}/injury-risk`),

  // Teams & Leagues
  getTeams: () => fetchJSON('/api/teams'),
  getLeagues: () => fetchJSON('/api/leagues'),

  // ML
  getClusters: () => fetchJSON('/api/clusters'),
  getMatchPrediction: (homeId, awayId) =>
    fetchJSON(`/api/predict-match?home_team_id=${homeId}&away_team_id=${awayId}`),
  getMLMetrics: () => fetchJSON('/api/ml-metrics'),

  // Analytics
  getAnomalies: () => fetchJSON('/api/anomalies'),
  getFatigueReport: () => fetchJSON('/api/fatigue-report'),
  getDecliningPlayers: () => fetchJSON('/api/declining-players'),
  getImprovingPlayers: () => fetchJSON('/api/improving-players'),
  getLineup: (team, formation) =>
    fetchJSON(`/api/lineup?team=${encodeURIComponent(team || '')}&formation=${formation || '4-3-3'}`),

  // Dashboards
  getCoachDashboard: () => fetchJSON('/dashboard/coach'),
  getScoutDashboard: () => fetchJSON('/dashboard/scout'),
  getAnalystDashboard: () => fetchJSON('/dashboard/analyst'),

  // Charts
  getPlayerRadar: (id) => fetchJSON(`/api/charts/player/${id}/radar`),
  getPlayerTrend: (id) => fetchJSON(`/api/charts/player/${id}/trend`),
  getFatigueChart: () => fetchJSON('/api/charts/fatigue'),
  getInjuryRiskChart: () => fetchJSON('/api/charts/injury-risk'),

  // Chat
  chat: async (message, history = []) => {
    const res = await fetch(`${BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history })
    });
    return res.json();
  }
};
