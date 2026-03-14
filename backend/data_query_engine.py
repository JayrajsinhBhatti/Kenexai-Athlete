"""
Data Query Engine — Natural Language to SQL + Data Retrieval
Converts user questions into SQL queries and retrieves structured data
"""
import sqlite3
import pandas as pd
import re
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database.sqlite')

# Pattern → SQL template mappings
QUERY_PATTERNS = [
    {
        'patterns': [r'top (\d+) players?(?: by| with highest| with best)? (.+)',
                     r'best (\d+) players?(?: by| in)? (.+)',
                     r'highest (\d+) (.+) players?'],
        'handler': 'top_players_by_attr'
    },
    {
        'patterns': [r'(?:players?|who) (?:from|in|plays? for) (.+)',
                     r'(.+) (?:players|squad|roster)'],
        'handler': 'players_from_team'
    },
    {
        'patterns': [r'(?:how many|count|total) (?:players|matches|teams|goals)',
                     r'(?:number of) (?:players|matches|teams|goals)'],
        'handler': 'count_query'
    },
    {
        'patterns': [r'(?:average|avg|mean) (.+)',
                     r'what is the average (.+)'],
        'handler': 'average_query'
    },
    {
        'patterns': [r'matches? (?:between|of|for) (.+?) (?:vs?|versus|against|and) (.+)',
                     r'(.+?) (?:vs?|versus|against) (.+?) (?:matches?|games?|results?)'],
        'handler': 'head_to_head'
    },
]

ATTR_MAP = {
    'rating': 'overall_rating', 'overall': 'overall_rating', 'overall rating': 'overall_rating',
    'potential': 'potential', 'speed': 'sprint_speed', 'sprint speed': 'sprint_speed',
    'stamina': 'stamina', 'strength': 'strength', 'acceleration': 'acceleration',
    'finishing': 'finishing', 'dribbling': 'dribbling', 'passing': 'short_passing',
    'shooting': 'finishing', 'defending': 'standing_tackle', 'crossing': 'crossing',
    'heading': 'heading_accuracy', 'vision': 'vision', 'agility': 'agility',
    'balance': 'balance', 'reactions': 'reactions', 'aggression': 'aggression',
    'attack': 'finishing', 'defense': 'standing_tackle', 'physical': 'strength',
    'performance': 'overall_rating', 'age': 'age', 'height': 'height', 'weight': 'weight',
}


class DataQueryEngine:
    def __init__(self, etl):
        self.etl = etl

    def query(self, question):
        """Convert natural language to data retrieval"""
        q = question.lower().strip()

        # Try pattern matching
        for pattern_group in QUERY_PATTERNS:
            for pattern in pattern_group['patterns']:
                match = re.search(pattern, q)
                if match:
                    handler = getattr(self, pattern_group['handler'], None)
                    if handler:
                        return handler(match, q)

        # Fallback: search for player/team names
        return self.smart_search(q)

    def top_players_by_attr(self, match, q):
        groups = match.groups()
        n = int(groups[0]) if groups[0].isdigit() else 10
        attr_text = groups[1].strip().rstrip('?.,!') if len(groups) > 1 else 'overall_rating'

        attr = ATTR_MAP.get(attr_text, 'overall_rating')
        perf = self.etl.gold['fact_player_performance']

        if attr not in perf.columns:
            attr = 'overall_rating'

        top = perf.nlargest(n, attr)[
            ['player_name', 'age', 'overall_rating', 'potential', 'performance_score', attr]
        ].to_dict('records')

        return {
            'type': 'player_ranking',
            'data': self._clean(top),
            'metric': attr,
            'count': n,
            'sql': f"SELECT player_name, {attr} FROM fact_player_performance ORDER BY {attr} DESC LIMIT {n}"
        }

    def players_from_team(self, match, q):
        team_name = match.group(1).strip().rstrip('?.,!')
        teams = self.etl.gold['dim_team']
        team = teams[teams['team_long_name'].str.lower().str.contains(team_name.lower(), na=False)]

        if team.empty:
            return {'type': 'error', 'data': None, 'message': f"Team '{team_name}' not found"}

        team_id = team.iloc[0]['team_api_id']
        team_full = team.iloc[0]['team_long_name']

        # Find players who played for this team in matches
        matches = self.etl.silver['Match']
        player_cols = [f'home_player_{i}' for i in range(1, 12)] + [f'away_player_{i}' for i in range(1, 12)]
        home_matches = matches[matches['home_team_api_id'] == team_id]
        away_matches = matches[matches['away_team_api_id'] == team_id]

        player_ids = set()
        for col in [f'home_player_{i}' for i in range(1, 12)]:
            if col in home_matches.columns:
                player_ids.update(home_matches[col].dropna().astype(int).tolist())
        for col in [f'away_player_{i}' for i in range(1, 12)]:
            if col in away_matches.columns:
                player_ids.update(away_matches[col].dropna().astype(int).tolist())

        perf = self.etl.gold['fact_player_performance']
        team_players = perf[perf['player_api_id'].isin(player_ids)].nlargest(20, 'performance_score')[
            ['player_name', 'age', 'overall_rating', 'potential', 'performance_score']
        ].to_dict('records')

        return {
            'type': 'team_players',
            'data': self._clean(team_players),
            'team': team_full,
            'count': len(team_players)
        }

    def count_query(self, match, q):
        perf = self.etl.gold['fact_player_performance']
        matches_df = self.etl.gold['fact_match']
        teams = self.etl.gold['dim_team']

        counts = {
            'total_players': len(perf),
            'total_matches': len(matches_df),
            'total_teams': len(teams),
            'total_goals': int(matches_df['total_goals'].sum()),
            'total_seasons': int(matches_df['season'].nunique()),
        }
        return {'type': 'counts', 'data': counts}

    def average_query(self, match, q):
        attr_text = match.group(1).strip().rstrip('?.,!')
        attr = ATTR_MAP.get(attr_text, 'overall_rating')
        perf = self.etl.gold['fact_player_performance']

        if attr not in perf.columns:
            attr = 'overall_rating'

        avg = round(float(perf[attr].mean()), 2)
        median = round(float(perf[attr].median()), 2)
        std = round(float(perf[attr].std()), 2)

        return {
            'type': 'statistics',
            'data': {'attribute': attr, 'mean': avg, 'median': median, 'std': std,
                     'min': round(float(perf[attr].min()), 2),
                     'max': round(float(perf[attr].max()), 2)},
            'sql': f"SELECT AVG({attr}), MEDIAN({attr}) FROM fact_player_performance"
        }

    def head_to_head(self, match, q):
        team1_name = match.group(1).strip()
        team2_name = match.group(2).strip()
        teams = self.etl.gold['dim_team']
        matches = self.etl.gold['fact_match']

        t1 = teams[teams['team_long_name'].str.lower().str.contains(team1_name.lower(), na=False)]
        t2 = teams[teams['team_long_name'].str.lower().str.contains(team2_name.lower(), na=False)]

        if t1.empty or t2.empty:
            return {'type': 'error', 'data': None, 'message': 'One or both teams not found'}

        t1_id, t2_id = t1.iloc[0]['team_api_id'], t2.iloc[0]['team_api_id']
        t1_name, t2_name = t1.iloc[0]['team_long_name'], t2.iloc[0]['team_long_name']

        h2h = matches[
            ((matches['home_team_api_id'] == t1_id) & (matches['away_team_api_id'] == t2_id)) |
            ((matches['home_team_api_id'] == t2_id) & (matches['away_team_api_id'] == t1_id))
        ].sort_values('date', ascending=False)

        results = h2h[['season', 'date', 'home_team_api_id', 'away_team_api_id',
                        'home_team_goal', 'away_team_goal', 'result']].head(10).to_dict('records')

        t1_wins = len(h2h[
            ((h2h['home_team_api_id'] == t1_id) & (h2h['result'] == 'Home Win')) |
            ((h2h['away_team_api_id'] == t1_id) & (h2h['result'] == 'Away Win'))
        ])
        t2_wins = len(h2h[
            ((h2h['home_team_api_id'] == t2_id) & (h2h['result'] == 'Home Win')) |
            ((h2h['away_team_api_id'] == t2_id) & (h2h['result'] == 'Away Win'))
        ])
        draws = len(h2h[h2h['result'] == 'Draw'])

        return {
            'type': 'head_to_head',
            'data': {
                'team1': t1_name, 'team2': t2_name,
                'total_matches': len(h2h),
                'team1_wins': t1_wins, 'team2_wins': t2_wins, 'draws': draws,
                'recent_matches': self._clean(results)
            }
        }

    def smart_search(self, q):
        """Fallback: search players and teams by name"""
        perf = self.etl.gold['fact_player_performance']
        teams = self.etl.gold['dim_team']

        # Extract potential names
        words = q.replace('?', '').replace('.', '').split()
        name_candidates = [w for w in words if len(w) > 2 and w[0].isupper()] or \
                          [w for w in words if len(w) > 3]

        for candidate in name_candidates:
            players = perf[perf['player_name'].str.lower().str.contains(candidate.lower(), na=False)]
            if not players.empty:
                return {
                    'type': 'player_search',
                    'data': self._clean(players.head(10)[
                        ['player_api_id', 'player_name', 'age', 'overall_rating', 'performance_score']
                    ].to_dict('records')),
                    'count': len(players)
                }

            team_matches = teams[teams['team_long_name'].str.lower().str.contains(candidate.lower(), na=False)]
            if not team_matches.empty:
                return {
                    'type': 'team_search',
                    'data': self._clean(team_matches[['team_api_id', 'team_long_name']].to_dict('records')),
                    'count': len(team_matches)
                }

        return {'type': 'no_results', 'data': None}

    def get_player_data(self, player_api_id):
        """Get full player data for analysis"""
        perf = self.etl.gold['fact_player_performance']
        player = perf[perf['player_api_id'] == player_api_id]
        if player.empty:
            return None
        return player.iloc[0].to_dict()

    def get_player_history(self, player_api_id):
        """Get player attribute history over time"""
        pa = self.etl.silver['Player_Attributes']
        history = pa[pa['player_api_id'] == player_api_id].sort_values('date')
        if history.empty:
            return None
        return history[['date', 'overall_rating', 'potential', 'stamina', 'strength',
                        'sprint_speed', 'acceleration', 'reactions']].to_dict('records')

    def _clean(self, records):
        import numpy as np
        cleaned = []
        for r in records:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (np.integer, np.int64)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    clean[k] = round(float(v), 2)
                elif pd.isna(v) if isinstance(v, float) else False:
                    clean[k] = None
                else:
                    clean[k] = v
            cleaned.append(clean)
        return cleaned
