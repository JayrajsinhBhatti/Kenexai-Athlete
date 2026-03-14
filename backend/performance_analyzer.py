"""
Performance Analyzer — Player trend detection and in-depth performance analysis
"""
import pandas as pd
import numpy as np


class PerformanceAnalyzer:
    def __init__(self, etl):
        self.etl = etl

    def analyze_player(self, player_api_id):
        """Full performance analysis for a single player"""
        perf = self.etl.gold['fact_player_performance']
        player = perf[perf['player_api_id'] == player_api_id]
        if player.empty:
            return None

        p = player.iloc[0]
        history = self._get_history(player_api_id)
        trend = self._compute_trend(history)
        strengths, weaknesses = self._find_strengths_weaknesses(p)
        position_fit = self._determine_position(p)
        percentiles = self._compute_percentiles(p, perf)

        return {
            'player': {
                'name': str(p['player_name']),
                'age': round(float(p['age']), 1) if pd.notna(p['age']) else None,
                'overall': round(float(p['overall_rating'])) if pd.notna(p['overall_rating']) else None,
                'potential': round(float(p['potential'])) if pd.notna(p['potential']) else None,
                'performance_score': round(float(p['performance_score']), 1) if pd.notna(p['performance_score']) else None,
            },
            'scores': {
                'attack': round(float(p['attack_score']), 1) if pd.notna(p['attack_score']) else 0,
                'midfield': round(float(p['midfield_score']), 1) if pd.notna(p['midfield_score']) else 0,
                'defense': round(float(p['defense_score']), 1) if pd.notna(p['defense_score']) else 0,
                'physical': round(float(p['physical_score']), 1) if pd.notna(p['physical_score']) else 0,
            },
            'trend': trend,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'position_fit': position_fit,
            'percentiles': percentiles,
            'development_gap': round(float(p['potential'] - p['overall_rating']), 1) if pd.notna(p['potential']) and pd.notna(p['overall_rating']) else 0,
        }

    def find_declining_players(self, min_decline=3):
        """Find players whose ratings are declining over time"""
        pa = self.etl.silver['Player_Attributes']
        declining = []

        for pid in pa['player_api_id'].unique():
            history = pa[pa['player_api_id'] == pid].sort_values('date')
            if len(history) < 3:
                continue

            ratings = history['overall_rating'].dropna().values
            if len(ratings) < 3:
                continue

            recent = np.mean(ratings[-3:])
            earlier = np.mean(ratings[:3])
            change = recent - earlier

            if change < -min_decline:
                perf = self.etl.gold['fact_player_performance']
                player = perf[perf['player_api_id'] == pid]
                if not player.empty:
                    p = player.iloc[0]
                    declining.append({
                        'player_name': str(p['player_name']),
                        'player_api_id': int(pid),
                        'current_rating': round(float(recent), 1),
                        'previous_rating': round(float(earlier), 1),
                        'decline': round(float(abs(change)), 1),
                        'data_points': len(ratings)
                    })

        declining.sort(key=lambda x: x['decline'], reverse=True)
        return declining[:20]

    def find_improving_players(self, min_improve=3):
        """Find players whose ratings are improving"""
        pa = self.etl.silver['Player_Attributes']
        improving = []

        for pid in pa['player_api_id'].unique():
            history = pa[pa['player_api_id'] == pid].sort_values('date')
            if len(history) < 3:
                continue

            ratings = history['overall_rating'].dropna().values
            if len(ratings) < 3:
                continue

            recent = np.mean(ratings[-3:])
            earlier = np.mean(ratings[:3])
            change = recent - earlier

            if change > min_improve:
                perf = self.etl.gold['fact_player_performance']
                player = perf[perf['player_api_id'] == pid]
                if not player.empty:
                    p = player.iloc[0]
                    improving.append({
                        'player_name': str(p['player_name']),
                        'player_api_id': int(pid),
                        'current_rating': round(float(recent), 1),
                        'previous_rating': round(float(earlier), 1),
                        'improvement': round(float(change), 1),
                    })

        improving.sort(key=lambda x: x['improvement'], reverse=True)
        return improving[:20]

    def compare_players(self, id1, id2):
        """In-depth comparison of two players"""
        a1 = self.analyze_player(id1)
        a2 = self.analyze_player(id2)
        if not a1 or not a2:
            return None

        attrs = ['attack', 'midfield', 'defense', 'physical']
        comparison = {}
        for attr in attrs:
            v1 = a1['scores'][attr]
            v2 = a2['scores'][attr]
            comparison[attr] = {
                'player1': v1, 'player2': v2,
                'difference': round(v1 - v2, 1),
                'winner': a1['player']['name'] if v1 > v2 else a2['player']['name'] if v2 > v1 else 'Tied'
            }

        overall_comparison = {
            'overall_rating': {
                'player1': a1['player']['overall'], 'player2': a2['player']['overall'],
                'winner': a1['player']['name'] if (a1['player']['overall'] or 0) > (a2['player']['overall'] or 0) else a2['player']['name']
            },
            'potential': {
                'player1': a1['player']['potential'], 'player2': a2['player']['potential'],
                'winner': a1['player']['name'] if (a1['player']['potential'] or 0) > (a2['player']['potential'] or 0) else a2['player']['name']
            },
            'performance': {
                'player1': a1['player']['performance_score'], 'player2': a2['player']['performance_score'],
                'winner': a1['player']['name'] if (a1['player']['performance_score'] or 0) > (a2['player']['performance_score'] or 0) else a2['player']['name']
            }
        }

        return {
            'player1': a1, 'player2': a2,
            'skill_comparison': comparison,
            'overall_comparison': overall_comparison,
        }

    def _get_history(self, player_api_id):
        pa = self.etl.silver['Player_Attributes']
        history = pa[pa['player_api_id'] == player_api_id].sort_values('date')
        return history

    def _compute_trend(self, history):
        if history.empty or len(history) < 2:
            return {'direction': 'stable', 'change': 0, 'data_points': 0}

        ratings = history['overall_rating'].dropna().values
        if len(ratings) < 2:
            return {'direction': 'stable', 'change': 0, 'data_points': len(ratings)}

        recent = np.mean(ratings[-3:]) if len(ratings) >= 3 else ratings[-1]
        earlier = np.mean(ratings[:3]) if len(ratings) >= 3 else ratings[0]
        change = round(float(recent - earlier), 1)

        direction = 'improving' if change > 1 else 'declining' if change < -1 else 'stable'

        trend_data = []
        for _, row in history.iterrows():
            if pd.notna(row.get('date')) and pd.notna(row.get('overall_rating')):
                trend_data.append({
                    'date': str(row['date'])[:10],
                    'rating': round(float(row['overall_rating']), 1),
                    'stamina': round(float(row['stamina']), 1) if pd.notna(row.get('stamina')) else None,
                })

        return {
            'direction': direction,
            'change': change,
            'data_points': len(ratings),
            'history': trend_data[-10:]
        }

    def _find_strengths_weaknesses(self, p):
        skill_map = {
            'Finishing': 'finishing', 'Heading': 'heading_accuracy',
            'Short Passing': 'short_passing', 'Long Passing': 'long_passing',
            'Dribbling': 'dribbling', 'Ball Control': 'ball_control',
            'Sprint Speed': 'sprint_speed', 'Acceleration': 'acceleration',
            'Stamina': 'stamina', 'Strength': 'strength',
            'Standing Tackle': 'standing_tackle', 'Marking': 'marking',
            'Vision': 'vision', 'Crossing': 'crossing',
            'Shot Power': 'shot_power', 'Agility': 'agility',
        }

        skills = {}
        for name, col in skill_map.items():
            val = p.get(col)
            if pd.notna(val):
                skills[name] = round(float(val), 1)

        sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
        strengths = [{'skill': s[0], 'value': s[1]} for s in sorted_skills[:5]]
        weaknesses = [{'skill': s[0], 'value': s[1]} for s in sorted_skills[-5:]]

        return strengths, weaknesses

    def _determine_position(self, p):
        scores = {
            'Forward': float(p['attack_score']) if pd.notna(p['attack_score']) else 0,
            'Midfielder': float(p['midfield_score']) if pd.notna(p['midfield_score']) else 0,
            'Defender': float(p['defense_score']) if pd.notna(p['defense_score']) else 0,
        }
        gk_skills = ['gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes']
        best_pos = max(scores, key=scores.get)

        return {
            'best_position': best_pos,
            'scores': {k: round(v, 1) for k, v in scores.items()},
            'versatility': 'High' if max(scores.values()) - min(scores.values()) < 15 else 'Specialized'
        }

    def _compute_percentiles(self, p, perf):
        result = {}
        for col in ['overall_rating', 'performance_score', 'attack_score', 'midfield_score', 'defense_score', 'physical_score']:
            val = p.get(col)
            if pd.notna(val):
                pct = round(float((perf[col].dropna() < val).mean() * 100), 1)
                result[col] = pct
        return result
