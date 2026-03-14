"""
AthleteIQ — Flask Backend API
Integrates ETL, ML Models, and GenAI Chat Agent
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from etl_pipeline import ETLPipeline
from ml_models import MLModels
from data_query_engine import DataQueryEngine
from performance_analyzer import PerformanceAnalyzer
from fatigue_monitor import FatigueMonitor, InjuryPredictor
from anomaly_detector import AnomalyDetector
from lineup_optimizer import LineupOptimizer
from chart_generator import ChartGenerator
from genai_chat_agent import GenAIChatAgent
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# ──────────── INITIALIZATION ────────────
print("Initializing backend...")
etl = ETLPipeline().run_full_pipeline()
ml = MLModels(etl).train_all()

# Initialize GenAI modules
data_engine = DataQueryEngine(etl)
perf_analyzer = PerformanceAnalyzer(etl)
fatigue_monitor = FatigueMonitor(etl)
injury_predictor = InjuryPredictor(etl, ml)
anomaly_detector = AnomalyDetector(etl)
lineup_optimizer = LineupOptimizer(etl, ml)
chart_generator = ChartGenerator(etl)

agent = GenAIChatAgent(
    etl, ml, data_engine, perf_analyzer, fatigue_monitor,
    injury_predictor, anomaly_detector, lineup_optimizer, chart_generator
)

print("Backend ready!\n")


# ──────────── HELPER ────────────
def clean_val(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 2) if not pd.isna(v) else None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def clean_dict(d):
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_dict(item) for item in d]
    else:
        return clean_val(d)


# ──────────── DATA QUALITY & EDA ────────────
@app.route('/api/data-quality')
def data_quality():
    return jsonify(etl.data_quality_report)


@app.route('/api/eda')
def eda():
    return jsonify(etl.get_eda_data())


# ──────────── PLAYERS ────────────
@app.route('/api/players')
def get_players():
    perf = etl.gold['fact_player_performance']
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 30))
    search = request.args.get('search', '')
    sort_by = request.args.get('sort_by', 'performance_score')

    filtered = perf.copy()
    if search:
        filtered = filtered[filtered['player_name'].str.lower().str.contains(search.lower(), na=False)]

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=False)

    total = len(filtered)
    start = (page - 1) * per_page
    page_data = filtered.iloc[start:start + per_page]

    cols = ['player_api_id', 'player_name', 'age', 'preferred_foot', 'overall_rating',
            'potential', 'performance_score', 'attack_score', 'midfield_score',
            'defense_score', 'physical_score']
    cols = [c for c in cols if c in page_data.columns]
    if 'cluster_name' in page_data.columns:
        cols.append('cluster_name')

    records = clean_dict(page_data[cols].to_dict('records'))
    return jsonify({'players': records, 'total': total, 'page': page, 'per_page': per_page})


@app.route('/api/players/<int:player_id>')
def get_player(player_id):
    perf = etl.gold['fact_player_performance']
    player = perf[perf['player_api_id'] == player_id]
    if player.empty:
        return jsonify({'error': 'Player not found'}), 404

    p = player.iloc[0]
    dim = etl.gold['dim_player']
    full = dim[dim['player_api_id'] == player_id]

    result = {
        'player_api_id': int(p['player_api_id']),
        'player_name': str(p['player_name']),
        'age': clean_val(p.get('age')),
        'height': clean_val(full.iloc[0].get('height') if not full.empty else None),
        'weight': clean_val(full.iloc[0].get('weight') if not full.empty else None),
        'preferred_foot': str(p.get('preferred_foot', '')),
        'overall_rating': clean_val(p.get('overall_rating')),
        'potential': clean_val(p.get('potential')),
        'performance_score': clean_val(p.get('performance_score')),
        'attack_score': clean_val(p.get('attack_score')),
        'midfield_score': clean_val(p.get('midfield_score')),
        'defense_score': clean_val(p.get('defense_score')),
        'physical_score': clean_val(p.get('physical_score')),
        'cluster_name': str(p.get('cluster_name', '')),
    }

    # Detailed skills
    skill_cols = ['crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys',
                  'dribbling', 'curve', 'free_kick_accuracy', 'long_passing', 'ball_control',
                  'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance',
                  'shot_power', 'jumping', 'stamina', 'strength', 'long_shots',
                  'aggression', 'interceptions', 'positioning', 'vision', 'penalties',
                  'marking', 'standing_tackle', 'sliding_tackle']
    result['skills'] = {c: clean_val(p.get(c)) for c in skill_cols if c in p.index}

    # Injury risk
    result['injury_risk'] = ml.predict_injury_risk(player_id)

    return jsonify(clean_dict(result))


@app.route('/api/players/<int:player_id>/injury-risk')
def get_injury_risk(player_id):
    risk = injury_predictor.predict_with_explanation(player_id)
    if not risk:
        return jsonify({'error': 'Player not found'}), 404
    return jsonify(clean_dict(risk))


# ──────────── TEAMS & LEAGUES ────────────
@app.route('/api/teams')
def get_teams():
    teams = etl.gold['dim_team'][['team_api_id', 'team_long_name', 'team_short_name']].to_dict('records')
    return jsonify(clean_dict(teams))


@app.route('/api/leagues')
def get_leagues():
    leagues = etl.gold['dim_league']
    result = []
    for _, l in leagues.iterrows():
        result.append({
            'id': clean_val(l.get('id')),
            'league_name': str(l.get('league_name', l.get('name', ''))),
            'country_name': str(l.get('country_name', ''))
        })
    return jsonify(result)


# ──────────── ML MODEL ENDPOINTS ────────────
@app.route('/api/ml-metrics')
def ml_metrics():
    return jsonify(ml.metrics)


@app.route('/api/clusters')
def get_clusters():
    clusters = ml.get_cluster_data()
    return jsonify(clean_dict(clusters))


@app.route('/api/predict-match')
def predict_match():
    home = request.args.get('home_team_id')
    away = request.args.get('away_team_id')
    if not home or not away:
        return jsonify({'error': 'home_team_id and away_team_id required'}), 400
    result = ml.predict_match(int(home), int(away))
    return jsonify(clean_dict(result))


# ──────────── ANALYTICS ENDPOINTS ────────────
@app.route('/api/anomalies')
def get_anomalies():
    anomalies = anomaly_detector.detect_all_anomalies()
    return jsonify(clean_dict(anomalies))


@app.route('/api/fatigue-report')
def get_fatigue_report():
    report = fatigue_monitor.get_squad_fatigue_report()
    return jsonify(clean_dict(report))


@app.route('/api/declining-players')
def get_declining():
    return jsonify(clean_dict(perf_analyzer.find_declining_players()))


@app.route('/api/improving-players')
def get_improving():
    return jsonify(clean_dict(perf_analyzer.find_improving_players()))


@app.route('/api/lineup')
def get_lineup():
    team = request.args.get('team', None)
    formation = request.args.get('formation', '4-3-3')
    result = lineup_optimizer.recommend_lineup(team_name=team, formation=formation)
    return jsonify(clean_dict(result))


# ──────────── DASHBOARD AGGREGATION ────────────
@app.route('/dashboard/coach')
def coach_dashboard():
    perf = etl.gold['fact_player_performance']
    injury = etl.gold['fact_injury_risk']

    top = perf.nlargest(10, 'performance_score')[
        ['player_api_id', 'player_name', 'overall_rating', 'potential', 'performance_score',
         'attack_score', 'midfield_score', 'defense_score', 'physical_score']
    ].to_dict('records')

    high_risk = injury[injury['injury_risk'] == 1].merge(
        perf[['player_api_id', 'player_name', 'overall_rating']],
        on='player_api_id', how='left'
    ).sort_values('fatigue_index', ascending=False).head(10)

    injury_alerts = [{
        'player_name': str(r.get('player_name', '')),
        'fatigue_index': round(float(r['fatigue_index']), 2),
        'training_load': round(float(r['training_load']), 1),
        'injury_risk': 'High'
    } for _, r in high_risk.iterrows()]

    # Squad composition
    if 'cluster_name' in perf.columns:
        composition = perf['cluster_name'].dropna().value_counts().to_dict()
    else:
        composition = {}

    # Performance distribution
    perf_levels = pd.cut(perf['performance_score'].dropna(),
                         bins=[0, 50, 60, 70, 80, 100],
                         labels=['Below Avg', 'Average', 'Good', 'Very Good', 'Elite']).value_counts()
    distribution = {str(k): int(v) for k, v in perf_levels.items()}

    return jsonify(clean_dict({
        'top_performers': top,
        'injury_alerts': injury_alerts,
        'squad_composition': composition,
        'performance_distribution': distribution,
        'summary': {
            'total_players': len(perf),
            'avg_rating': round(float(perf['overall_rating'].mean()), 1),
            'avg_performance': round(float(perf['performance_score'].mean()), 1),
            'high_risk_count': int(injury['injury_risk'].sum()),
            'low_risk_count': int((injury['injury_risk'] == 0).sum()),
        }
    }))


@app.route('/dashboard/scout')
def scout_dashboard():
    perf = etl.gold['fact_player_performance']

    young = perf[(perf['age'] < 23) & (perf['potential'] > 70)].nlargest(15, 'potential')
    young_talents = young[['player_api_id', 'player_name', 'age', 'overall_rating', 'potential',
                           'performance_score', 'attack_score', 'midfield_score', 'defense_score']].to_dict('records')

    perf_copy = perf.copy()
    perf_copy['value_gap'] = perf_copy['performance_score'] - perf_copy['overall_rating']
    undervalued = perf_copy.nlargest(15, 'value_gap')[
        ['player_api_id', 'player_name', 'age', 'overall_rating', 'potential',
         'performance_score', 'value_gap']
    ].to_dict('records')

    # Player clusters for scout view
    clusters = []
    if 'cluster_name' in perf.columns:
        for name in perf['cluster_name'].dropna().unique():
            group = perf[perf['cluster_name'] == name]
            cluster_players = group.nlargest(5, 'performance_score')[
                ['player_name', 'overall_rating', 'performance_score']
            ].to_dict('records')
            clusters.append({
                'name': str(name),
                'count': len(group),
                'avg_rating': round(float(group['overall_rating'].mean()), 1),
                'center': {
                    'attack': round(float(group['attack_score'].mean()), 1),
                    'midfield': round(float(group['midfield_score'].mean()), 1),
                    'defense': round(float(group['defense_score'].mean()), 1),
                    'physical': round(float(group['physical_score'].mean()), 1),
                },
                'top_players': cluster_players
            })

    return jsonify(clean_dict({
        'young_talents': young_talents,
        'undervalued_players': undervalued,
        'player_clusters': clusters,
        'summary': {
            'total_scouted': len(perf),
            'young_talents_count': len(young),
            'avg_potential': round(float(perf['potential'].mean()), 1),
        }
    }))


@app.route('/dashboard/analyst')
def analyst_dashboard():
    matches = etl.gold['fact_match']
    perf = etl.gold['fact_player_performance']
    teams = etl.gold['dim_team']

    goals_by_season = matches.groupby('season').agg(
        total_goals=('total_goals', 'sum'),
        avg_goals=('total_goals', 'mean'),
        matches_count=('total_goals', 'count'),
        avg_home_goals=('home_team_goal', 'mean'),
        avg_away_goals=('away_team_goal', 'mean'),
    ).reset_index().to_dict('records')

    result_dist = matches['result'].value_counts().to_dict()

    # Top scoring teams
    home_goals = matches.groupby('home_team_api_id')['home_team_goal'].sum().reset_index()
    home_goals.columns = ['team_api_id', 'goals']
    away_goals = matches.groupby('away_team_api_id')['away_team_goal'].sum().reset_index()
    away_goals.columns = ['team_api_id', 'goals']
    all_goals = pd.concat([home_goals, away_goals]).groupby('team_api_id')['goals'].sum().reset_index()
    all_goals = all_goals.merge(teams[['team_api_id', 'team_long_name']], on='team_api_id', how='left')
    top_scoring = all_goals.nlargest(10, 'goals')[['team_long_name', 'goals']].to_dict('records')

    return jsonify(clean_dict({
        'summary': {
            'total_matches': len(matches),
            'total_goals': int(matches['total_goals'].sum()),
            'avg_goals_per_match': round(float(matches['total_goals'].mean()), 2),
            'teams': len(teams),
        },
        'goals_by_season': goals_by_season,
        'result_distribution': result_dist,
        'top_scoring_teams': top_scoring,
        'ml_performance': ml.metrics,
    }))


# ──────────── GenAI CHAT ────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Message required'}), 400

    user_message = data['message']
    response = agent.process_message(user_message)
    return jsonify(clean_dict(response))


# ──────────── CHART DATA ────────────
@app.route('/api/charts/player/<int:player_id>/radar')
def player_radar(player_id):
    chart = chart_generator.player_performance_radar(player_id)
    return jsonify(clean_dict(chart)) if chart else (jsonify({'error': 'Not found'}), 404)


@app.route('/api/charts/player/<int:player_id>/trend')
def player_trend(player_id):
    chart = chart_generator.performance_trend(player_id)
    return jsonify(clean_dict(chart)) if chart else (jsonify({'error': 'Not found'}), 404)


@app.route('/api/charts/fatigue')
def fatigue_chart():
    return jsonify(clean_dict(chart_generator.fatigue_distribution()))


@app.route('/api/charts/injury-risk')
def injury_risk_chart():
    return jsonify(clean_dict(chart_generator.injury_risk_chart()))


@app.route('/api/charts/goals-by-season')
def goals_season_chart():
    return jsonify(clean_dict(chart_generator.goals_by_season()))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
