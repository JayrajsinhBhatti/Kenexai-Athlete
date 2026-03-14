"""
Lineup Optimizer — Recommend optimal starting XI based on performance, fatigue, and injury risk
"""
import pandas as pd
import numpy as np


FORMATIONS = {
    '4-3-3': {'GK': 1, 'DEF': 4, 'MID': 3, 'FWD': 3},
    '4-4-2': {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2},
    '3-5-2': {'GK': 1, 'DEF': 3, 'MID': 5, 'FWD': 2},
    '4-2-3-1': {'GK': 1, 'DEF': 4, 'MID': 5, 'FWD': 1},
    '3-4-3': {'GK': 1, 'DEF': 3, 'MID': 4, 'FWD': 3},
}


class LineupOptimizer:
    def __init__(self, etl, ml):
        self.etl = etl
        self.ml = ml

    def recommend_lineup(self, team_name=None, formation='4-3-3', player_pool=None):
        """Recommend optimal starting XI"""
        # Get player pool
        if player_pool:
            players = self._get_player_pool(player_pool)
        elif team_name:
            players = self._get_team_players(team_name)
        else:
            # Use top players from entire dataset
            players = self._get_top_pool()

        if players.empty or len(players) < 11:
            return {'error': 'Not enough players to form a lineup'}

        # Add fitness scores
        players = self._add_fitness_data(players)

        # Classify positions
        players = self._classify_positions(players)

        # Optimize lineup
        formation_spec = FORMATIONS.get(formation, FORMATIONS['4-3-3'])
        lineup = self._select_lineup(players, formation_spec)

        # Calculate team chemistry
        chemistry = self._calculate_chemistry(lineup)

        return {
            'formation': formation,
            'lineup': lineup,
            'team_stats': self._team_stats(lineup),
            'chemistry': chemistry,
            'bench': self._select_bench(players, lineup),
        }

    def _get_team_players(self, team_name):
        teams = self.etl.gold['dim_team']
        team = teams[teams['team_long_name'].str.lower().str.contains(team_name.lower(), na=False)]
        if team.empty:
            return pd.DataFrame()

        team_id = team.iloc[0]['team_api_id']
        matches = self.etl.silver['Match']
        player_ids = set()

        home = matches[matches['home_team_api_id'] == team_id]
        for i in range(1, 12):
            col = f'home_player_{i}'
            if col in home.columns:
                player_ids.update(home[col].dropna().astype(int).tolist())

        away = matches[matches['away_team_api_id'] == team_id]
        for i in range(1, 12):
            col = f'away_player_{i}'
            if col in away.columns:
                player_ids.update(away[col].dropna().astype(int).tolist())

        perf = self.etl.gold['fact_player_performance']
        return perf[perf['player_api_id'].isin(player_ids)].copy()

    def _get_player_pool(self, player_ids):
        perf = self.etl.gold['fact_player_performance']
        return perf[perf['player_api_id'].isin(player_ids)].copy()

    def _get_top_pool(self):
        perf = self.etl.gold['fact_player_performance']
        return perf.nlargest(100, 'performance_score').copy()

    def _add_fitness_data(self, players):
        injury = self.etl.gold['fact_injury_risk']
        merged = players.merge(
            injury[['player_api_id', 'fatigue_index', 'injury_risk', 'training_load', 'sleep_hours']],
            on='player_api_id', how='left'
        )
        merged['fatigue_index'] = merged['fatigue_index'].fillna(5.0)
        merged['injury_risk'] = merged['injury_risk'].fillna(0)

        # Availability score: penalize injured/fatigued players
        merged['availability_score'] = (
            merged['performance_score'] *
            (1 - merged['injury_risk'] * 0.3) *
            (1 - (merged['fatigue_index'] / 10) * 0.2)
        )
        return merged

    def _classify_positions(self, players):
        def classify(row):
            atk = float(row['attack_score']) if pd.notna(row.get('attack_score')) else 0
            mid = float(row['midfield_score']) if pd.notna(row.get('midfield_score')) else 0
            dfn = float(row['defense_score']) if pd.notna(row.get('defense_score')) else 0

            # Check for goalkeeper
            gk_cols = ['gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes']
            is_gk = all(pd.notna(row.get(c)) and float(row.get(c, 0) or 0) > 50 for c in gk_cols if c in row.index)
            if is_gk and atk < 30 and mid < 40:
                return 'GK'

            if dfn > mid and dfn > atk:
                return 'DEF'
            if mid > atk and mid >= dfn:
                return 'MID'
            return 'FWD'

        players['position'] = players.apply(classify, axis=1)
        return players

    def _select_lineup(self, players, formation_spec):
        lineup = []
        used = set()

        for pos, count in formation_spec.items():
            candidates = players[
                (players['position'] == pos) &
                (~players['player_api_id'].isin(used))
            ].sort_values('availability_score', ascending=False)

            # If not enough players at this position, fill from others
            if len(candidates) < count:
                extra = players[
                    (~players['player_api_id'].isin(used)) &
                    (~players['player_api_id'].isin(candidates['player_api_id']))
                ].sort_values('availability_score', ascending=False)
                candidates = pd.concat([candidates, extra])

            selected = candidates.head(count)
            for _, p in selected.iterrows():
                used.add(p['player_api_id'])
                lineup.append({
                    'position': pos,
                    'player_name': str(p['player_name']),
                    'player_api_id': int(p['player_api_id']),
                    'overall_rating': round(float(p['overall_rating'])) if pd.notna(p['overall_rating']) else None,
                    'performance_score': round(float(p['performance_score']), 1) if pd.notna(p['performance_score']) else None,
                    'fatigue_index': round(float(p.get('fatigue_index', 5)), 2),
                    'injury_risk': 'High' if p.get('injury_risk', 0) == 1 else 'Low',
                    'fitness_status': '🟢 Fit' if p.get('fatigue_index', 5) < 5 else '🟡 Monitor' if p.get('fatigue_index', 5) < 7 else '🔴 At Risk',
                })

        return lineup

    def _select_bench(self, all_players, lineup):
        used_ids = {p['player_api_id'] for p in lineup}
        bench = all_players[~all_players['player_api_id'].isin(used_ids)].sort_values('availability_score', ascending=False).head(7)

        return [{
            'player_name': str(row['player_name']),
            'position': row['position'],
            'overall_rating': round(float(row['overall_rating'])) if pd.notna(row['overall_rating']) else None,
            'performance_score': round(float(row['performance_score']), 1) if pd.notna(row['performance_score']) else None,
        } for _, row in bench.iterrows()]

    def _team_stats(self, lineup):
        ratings = [p['overall_rating'] for p in lineup if p['overall_rating']]
        perfs = [p['performance_score'] for p in lineup if p['performance_score']]
        fatigues = [p['fatigue_index'] for p in lineup]
        high_risk = sum(1 for p in lineup if p['injury_risk'] == 'High')

        return {
            'avg_rating': round(np.mean(ratings), 1) if ratings else 0,
            'avg_performance': round(np.mean(perfs), 1) if perfs else 0,
            'avg_fatigue': round(np.mean(fatigues), 2),
            'high_risk_count': high_risk,
            'squad_fitness': 'Good' if np.mean(fatigues) < 5 else 'Concerning' if np.mean(fatigues) < 7 else 'Critical',
        }

    def _calculate_chemistry(self, lineup):
        positions = [p['position'] for p in lineup]
        ratings = [p['overall_rating'] or 65 for p in lineup]

        balance = 1.0 - (np.std(ratings) / np.mean(ratings)) if np.mean(ratings) > 0 else 0.5
        chemistry_score = round(balance * 100, 1)

        return {
            'score': min(95, chemistry_score),
            'level': 'Excellent' if chemistry_score > 85 else 'Good' if chemistry_score > 70 else 'Average',
        }
