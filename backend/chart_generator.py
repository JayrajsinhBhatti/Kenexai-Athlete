"""
Chart Generator — Generate chart data for the frontend to render
Returns structured JSON that the frontend can visualize with Recharts
"""
import pandas as pd
import numpy as np


class ChartGenerator:
    def __init__(self, etl):
        self.etl = etl

    def player_performance_radar(self, player_api_id):
        """Generate radar chart data for a player"""
        perf = self.etl.gold['fact_player_performance']
        player = perf[perf['player_api_id'] == player_api_id]
        if player.empty:
            return None

        p = player.iloc[0]
        return {
            'type': 'radar',
            'title': f"{p['player_name']} - Skill Profile",
            'data': [
                {'skill': 'Attack', 'value': round(float(p['attack_score']), 1) if pd.notna(p['attack_score']) else 0, 'fullMark': 100},
                {'skill': 'Midfield', 'value': round(float(p['midfield_score']), 1) if pd.notna(p['midfield_score']) else 0, 'fullMark': 100},
                {'skill': 'Defense', 'value': round(float(p['defense_score']), 1) if pd.notna(p['defense_score']) else 0, 'fullMark': 100},
                {'skill': 'Physical', 'value': round(float(p['physical_score']), 1) if pd.notna(p['physical_score']) else 0, 'fullMark': 100},
                {'skill': 'Overall', 'value': round(float(p['overall_rating']), 1) if pd.notna(p['overall_rating']) else 0, 'fullMark': 100},
                {'skill': 'Potential', 'value': round(float(p['potential']), 1) if pd.notna(p['potential']) else 0, 'fullMark': 100},
            ]
        }

    def performance_trend(self, player_api_id):
        """Generate performance trend line chart data"""
        pa = self.etl.silver['Player_Attributes']
        history = pa[pa['player_api_id'] == player_api_id].sort_values('date')
        if history.empty:
            return None

        perf = self.etl.gold['fact_player_performance']
        player = perf[perf['player_api_id'] == player_api_id]
        name = str(player.iloc[0]['player_name']) if not player.empty else 'Unknown'

        data = []
        for _, row in history.iterrows():
            if pd.notna(row.get('date')):
                data.append({
                    'date': str(row['date'])[:10],
                    'overall_rating': round(float(row['overall_rating']), 1) if pd.notna(row.get('overall_rating')) else None,
                    'potential': round(float(row['potential']), 1) if pd.notna(row.get('potential')) else None,
                    'stamina': round(float(row['stamina']), 1) if pd.notna(row.get('stamina')) else None,
                })

        return {'type': 'line', 'title': f"{name} - Performance Trend", 'data': data}

    def fatigue_distribution(self):
        """Generate fatigue distribution chart data"""
        injury = self.etl.gold['fact_injury_risk']
        perf = self.etl.gold['fact_player_performance']

        merged = injury.merge(perf[['player_api_id', 'player_name']], on='player_api_id', how='left')
        merged['fatigue_level'] = pd.cut(merged['fatigue_index'],
                                          bins=[0, 3, 5, 7, 10],
                                          labels=['Low', 'Moderate', 'High', 'Critical'])
        dist = merged['fatigue_level'].value_counts()

        return {
            'type': 'pie',
            'title': 'Squad Fatigue Distribution',
            'data': [{'name': str(k), 'value': int(v)} for k, v in dist.items()]
        }

    def injury_risk_chart(self):
        """Generate injury risk breakdown"""
        injury = self.etl.gold['fact_injury_risk']
        high = int(injury['injury_risk'].sum())
        low = int((injury['injury_risk'] == 0).sum())

        return {
            'type': 'pie',
            'title': 'Injury Risk Distribution',
            'data': [
                {'name': 'High Risk', 'value': high, 'color': '#ef4444'},
                {'name': 'Low Risk', 'value': low, 'color': '#10b981'},
            ]
        }

    def goals_by_season(self):
        """Goals per season trend"""
        matches = self.etl.gold['fact_match']
        by_season = matches.groupby('season').agg(
            total_goals=('total_goals', 'sum'),
            avg_goals=('total_goals', 'mean'),
            matches_count=('total_goals', 'count')
        ).reset_index()

        data = []
        for _, row in by_season.iterrows():
            data.append({
                'season': str(row['season']),
                'total_goals': int(row['total_goals']),
                'avg_goals': round(float(row['avg_goals']), 2),
                'matches': int(row['matches_count']),
            })

        return {'type': 'bar', 'title': 'Goals by Season', 'data': data}

    def player_comparison_chart(self, id1, id2):
        """Generate comparison chart data for two players"""
        perf = self.etl.gold['fact_player_performance']
        p1 = perf[perf['player_api_id'] == id1]
        p2 = perf[perf['player_api_id'] == id2]

        if p1.empty or p2.empty:
            return None

        p1, p2 = p1.iloc[0], p2.iloc[0]
        attrs = ['attack_score', 'midfield_score', 'defense_score', 'physical_score', 'overall_rating', 'potential']
        data = []
        for attr in attrs:
            v1 = round(float(p1[attr]), 1) if pd.notna(p1[attr]) else 0
            v2 = round(float(p2[attr]), 1) if pd.notna(p2[attr]) else 0
            data.append({
                'skill': attr.replace('_', ' ').title().replace(' Score', ''),
                'player1': v1,
                'player2': v2,
            })

        return {
            'type': 'radar_comparison',
            'title': f"{p1['player_name']} vs {p2['player_name']}",
            'player1_name': str(p1['player_name']),
            'player2_name': str(p2['player_name']),
            'data': data
        }

    def top_players_chart(self, metric='performance_score', n=10):
        """Bar chart of top N players by metric"""
        perf = self.etl.gold['fact_player_performance']
        if metric not in perf.columns:
            metric = 'performance_score'

        top = perf.nlargest(n, metric)
        data = []
        for _, row in top.iterrows():
            data.append({
                'player_name': str(row['player_name']),
                'value': round(float(row[metric]), 1) if pd.notna(row[metric]) else 0,
            })

        return {'type': 'bar', 'title': f'Top {n} Players by {metric.replace("_", " ").title()}', 'data': data}
