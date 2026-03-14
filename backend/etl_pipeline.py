"""
ETL Pipeline - Medallion Architecture (Bronze → Silver → Gold)
Athlete Performance Analytics & Injury Risk Prediction
"""
import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import os
import json
from datetime import datetime, timedelta
import random

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database.sqlite')

class ETLPipeline:
    def __init__(self):
        self.bronze = {}
        self.silver = {}
        self.gold = {}
        self.data_quality_report = {}

    # ──────────────── BRONZE LAYER ────────────────
    def extract_bronze(self):
        """Load raw data from SQLite into DataFrames"""
        conn = sqlite3.connect(DB_PATH)
        tables = ['Player', 'Player_Attributes', 'Match', 'Team', 'Team_Attributes', 'League', 'Country']
        for table in tables:
            self.bronze[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
        print(f"[Bronze] Loaded {len(tables)} tables from SQLite")
        return self

    # ──────────────── SILVER LAYER ────────────────
    def transform_silver(self):
        """Clean, type-convert, handle missing values, detect outliers"""
        # --- Players ---
        players = self.bronze['Player'].copy()
        players['birthday'] = pd.to_datetime(players['birthday'], errors='coerce')
        players['age'] = ((pd.Timestamp.now() - players['birthday']).dt.days / 365.25).round(1)
        players['height'] = pd.to_numeric(players['height'], errors='coerce')
        players['weight'] = pd.to_numeric(players['weight'], errors='coerce')
        players['height'].fillna(players['height'].median(), inplace=True)
        players['weight'].fillna(players['weight'].median(), inplace=True)
        self.silver['Player'] = players

        # --- Player Attributes ---
        pa = self.bronze['Player_Attributes'].copy()
        pa['date'] = pd.to_datetime(pa['date'], errors='coerce')
        numeric_cols = pa.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['id', 'player_fifa_api_id', 'player_api_id']]
        for col in numeric_cols:
            pa[col] = pd.to_numeric(pa[col], errors='coerce')
            pa[col].fillna(pa[col].median(), inplace=True)
        pa['preferred_foot'].fillna('right', inplace=True)
        pa['attacking_work_rate'].fillna('medium', inplace=True)
        pa['defensive_work_rate'].fillna('medium', inplace=True)
        # Outlier detection (IQR)
        outlier_info = {}
        for col in numeric_cols:
            Q1 = pa[col].quantile(0.25)
            Q3 = pa[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count = ((pa[col] < lower) | (pa[col] > upper)).sum()
            outlier_info[col] = {'lower': round(lower, 2), 'upper': round(upper, 2), 'outliers': int(outlier_count)}
            pa[col] = pa[col].clip(lower=lower, upper=upper)
        self.data_quality_report['outliers'] = outlier_info
        self.silver['Player_Attributes'] = pa

        # --- Matches ---
        matches = self.bronze['Match'].copy()
        matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
        matches['home_team_goal'] = pd.to_numeric(matches['home_team_goal'], errors='coerce').fillna(0).astype(int)
        matches['away_team_goal'] = pd.to_numeric(matches['away_team_goal'], errors='coerce').fillna(0).astype(int)
        matches['total_goals'] = matches['home_team_goal'] + matches['away_team_goal']
        matches['goal_difference'] = matches['home_team_goal'] - matches['away_team_goal']
        matches['result'] = matches['goal_difference'].apply(lambda x: 'Home Win' if x > 0 else ('Draw' if x == 0 else 'Away Win'))
        self.silver['Match'] = matches

        # --- Teams ---
        teams = self.bronze['Team'].copy()
        self.silver['Team'] = teams

        # --- Team Attributes ---
        ta = self.bronze['Team_Attributes'].copy()
        ta['date'] = pd.to_datetime(ta['date'], errors='coerce')
        ta_numeric = ta.select_dtypes(include=[np.number]).columns.tolist()
        ta_numeric = [c for c in ta_numeric if c not in ['id', 'team_fifa_api_id', 'team_api_id']]
        for col in ta_numeric:
            ta[col] = pd.to_numeric(ta[col], errors='coerce')
            ta[col].fillna(ta[col].median(), inplace=True)
        self.silver['Team_Attributes'] = ta

        # --- League & Country ---
        self.silver['League'] = self.bronze['League'].copy()
        self.silver['Country'] = self.bronze['Country'].copy()

        print("[Silver] Data cleaned and transformed")
        return self

    # ──────────────── GOLD LAYER ────────────────
    def load_gold(self):
        """Create denormalized analytics-ready tables"""

        # --- dim_player ---
        players = self.silver['Player']
        latest_attrs = self.silver['Player_Attributes'].sort_values('date', ascending=False).drop_duplicates('player_api_id')
        dim_player = players.merge(latest_attrs, on='player_api_id', how='left', suffixes=('', '_attr'))
        dim_player = dim_player.rename(columns={'id': 'player_id'})
        self.gold['dim_player'] = dim_player

        # --- dim_team ---
        teams = self.silver['Team']
        latest_ta = self.silver['Team_Attributes'].sort_values('date', ascending=False).drop_duplicates('team_api_id')
        dim_team = teams.merge(latest_ta, on='team_api_id', how='left', suffixes=('', '_attr'))
        self.gold['dim_team'] = dim_team

        # --- dim_league ---
        leagues = self.silver['League'].merge(self.silver['Country'], left_on='country_id', right_on='id', how='left', suffixes=('', '_country'))
        leagues = leagues.rename(columns={'name': 'league_name', 'name_country': 'country_name'})
        self.gold['dim_league'] = leagues

        # --- fact_match_results ---
        matches = self.silver['Match']
        fact_match = matches[['id', 'country_id', 'league_id', 'season', 'stage', 'date',
                              'home_team_api_id', 'away_team_api_id',
                              'home_team_goal', 'away_team_goal', 'total_goals',
                              'goal_difference', 'result']].copy()
        fact_match = fact_match.rename(columns={'id': 'match_id'})
        self.gold['fact_match'] = fact_match

        # --- fact_player_performance ---
        skill_cols = ['overall_rating', 'potential', 'crossing', 'finishing', 'heading_accuracy',
                      'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
                      'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility',
                      'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 'strength',
                      'long_shots', 'aggression', 'interceptions', 'positioning', 'vision',
                      'penalties', 'marking', 'standing_tackle', 'sliding_tackle']
        perf = dim_player[['player_api_id', 'player_name', 'age', 'height', 'weight',
                           'preferred_foot'] + skill_cols].copy()

        # Compute composite performance score
        attack_cols = ['finishing', 'heading_accuracy', 'volleys', 'shot_power', 'long_shots', 'positioning']
        midfield_cols = ['short_passing', 'long_passing', 'ball_control', 'dribbling', 'vision', 'crossing', 'curve']
        defense_cols = ['marking', 'standing_tackle', 'sliding_tackle', 'interceptions', 'aggression']
        physical_cols = ['acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'stamina', 'strength', 'jumping']

        perf['attack_score'] = perf[attack_cols].mean(axis=1)
        perf['midfield_score'] = perf[midfield_cols].mean(axis=1)
        perf['defense_score'] = perf[defense_cols].mean(axis=1)
        perf['physical_score'] = perf[physical_cols].mean(axis=1)
        perf['performance_score'] = (perf['overall_rating'] * 0.35 +
                                     perf['potential'] * 0.15 +
                                     perf['attack_score'] * 0.15 +
                                     perf['midfield_score'] * 0.15 +
                                     perf['defense_score'] * 0.10 +
                                     perf['physical_score'] * 0.10)
        perf['performance_score'] = perf['performance_score'].round(1)
        self.gold['fact_player_performance'] = perf

        # --- Generate synthetic wearable/injury data ---
        self._generate_injury_data()

        print("[Gold] Analytics-ready tables created")
        return self

    def _generate_injury_data(self):
        """Generate synthetic wearable sensor data for injury risk modeling"""
        np.random.seed(42)
        random.seed(42)
        players = self.gold['dim_player']
        records = []
        for _, p in players.iterrows():
            age = p.get('age', 25) if pd.notna(p.get('age')) else 25
            stamina = p.get('stamina', 60) if pd.notna(p.get('stamina')) else 60
            strength = p.get('strength', 60) if pd.notna(p.get('strength')) else 60
            sprint_speed = p.get('sprint_speed', 60) if pd.notna(p.get('sprint_speed')) else 60

            # Simulate wearable data with correlations to player attributes
            avg_heart_rate = max(55, min(95, np.random.normal(72 + (age - 25) * 0.5, 8)))
            max_heart_rate = max(160, min(210, np.random.normal(200 - age * 0.5, 10)))
            sprint_distance = max(0.5, np.random.normal(sprint_speed / 10, 1.5))
            total_distance = max(5, np.random.normal(9 + stamina / 30, 1.5))
            training_load = max(200, np.random.normal(500 + (100 - stamina) * 2, 80))
            fatigue_index = max(1, min(10, np.random.normal(5 + (age - 25) * 0.15 + (100 - stamina) * 0.03, 1.5)))
            sleep_hours = max(4, min(10, np.random.normal(7.5 - (age - 25) * 0.05, 1)))
            matches_last_30d = max(0, int(np.random.normal(4, 1.5)))
            previous_injuries = max(0, int(np.random.exponential(1.5 + (age - 25) * 0.1)))
            recovery_time = max(12, np.random.normal(36 + (age - 25) * 0.8, 8))

            # Injury risk based on realistic factors
            risk_score = (
                (age - 20) * 0.08 +
                (100 - stamina) * 0.03 +
                (100 - strength) * 0.02 +
                fatigue_index * 0.12 +
                training_load / 500 * 0.15 +
                matches_last_30d * 0.08 +
                previous_injuries * 0.15 +
                (10 - sleep_hours) * 0.1 +
                np.random.normal(0, 0.1)
            )
            injury_risk = 1 if risk_score > 5.3 else 0

            records.append({
                'player_api_id': p['player_api_id'],
                'avg_heart_rate': round(avg_heart_rate, 1),
                'max_heart_rate': round(max_heart_rate, 1),
                'sprint_distance_km': round(sprint_distance, 2),
                'total_distance_km': round(total_distance, 2),
                'training_load': round(training_load, 1),
                'fatigue_index': round(fatigue_index, 2),
                'sleep_hours': round(sleep_hours, 1),
                'matches_last_30d': matches_last_30d,
                'previous_injuries': previous_injuries,
                'recovery_time_hours': round(recovery_time, 1),
                'injury_risk': injury_risk
            })

        self.gold['fact_injury_risk'] = pd.DataFrame(records)

    # ──────────────── DATA QUALITY ────────────────
    def compute_data_quality(self):
        """Compute comprehensive data quality metrics"""
        report = {}
        for name, df in self.bronze.items():
            total = df.shape[0] * df.shape[1]
            missing = df.isnull().sum().sum()
            duplicates = df.duplicated().sum()

            col_quality = {}
            for col in df.columns:
                col_missing = int(df[col].isnull().sum())
                col_total = len(df[col])
                col_quality[col] = {
                    'missing': col_missing,
                    'completeness': round((1 - col_missing / col_total) * 100, 1) if col_total > 0 else 100,
                    'dtype': str(df[col].dtype),
                    'unique': int(df[col].nunique())
                }

            report[name] = {
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1]),
                'total_cells': int(total),
                'missing_cells': int(missing),
                'completeness': round((1 - missing / total) * 100, 1) if total > 0 else 100,
                'duplicates': int(duplicates),
                'column_quality': col_quality
            }

        report['outliers'] = self.data_quality_report.get('outliers', {})
        self.data_quality_report = report
        print("[Quality] Data quality report generated")
        return self

    def get_eda_data(self):
        """Generate EDA statistics and chart-ready data"""
        pa = self.gold['fact_player_performance']
        eda = {}

        # Distribution of overall ratings
        eda['rating_distribution'] = pa['overall_rating'].dropna().value_counts().sort_index().to_dict()
        eda['rating_distribution'] = {str(int(k)): int(v) for k, v in eda['rating_distribution'].items()}

        # Performance score distribution
        bins = list(range(0, 105, 5))
        perf_hist = pd.cut(pa['performance_score'].dropna(), bins=bins).value_counts().sort_index()
        eda['performance_distribution'] = {str(interval): int(count) for interval, count in perf_hist.items()}

        # Age distribution
        age_bins = list(range(15, 50, 2))
        age_hist = pd.cut(pa['age'].dropna(), bins=age_bins).value_counts().sort_index()
        eda['age_distribution'] = {str(interval): int(count) for interval, count in age_hist.items()}

        # Preferred foot
        eda['preferred_foot'] = pa['preferred_foot'].value_counts().to_dict()

        # Correlation matrix for key attributes
        corr_cols = ['overall_rating', 'potential', 'attack_score', 'midfield_score',
                     'defense_score', 'physical_score', 'performance_score', 'age']
        corr_df = pa[corr_cols].dropna()
        corr_matrix = corr_df.corr().round(3)
        eda['correlation_matrix'] = {
            'columns': corr_cols,
            'values': corr_matrix.values.tolist()
        }

        # Top players
        top_players = pa.nlargest(20, 'performance_score')[['player_name', 'overall_rating', 'potential', 'performance_score', 'age']].to_dict('records')
        eda['top_players'] = top_players

        # Summary stats
        eda['summary_stats'] = {
            'total_players': int(len(pa)),
            'avg_rating': round(float(pa['overall_rating'].mean()), 1),
            'avg_age': round(float(pa['age'].mean()), 1),
            'avg_performance': round(float(pa['performance_score'].mean()), 1),
            'max_performance': round(float(pa['performance_score'].max()), 1),
            'min_performance': round(float(pa['performance_score'].min()), 1)
        }

        # Match stats
        matches = self.gold['fact_match']
        eda['match_stats'] = {
            'total_matches': int(len(matches)),
            'avg_goals_per_match': round(float(matches['total_goals'].mean()), 2),
            'home_win_pct': round(float((matches['result'] == 'Home Win').mean() * 100), 1),
            'draw_pct': round(float((matches['result'] == 'Draw').mean() * 100), 1),
            'away_win_pct': round(float((matches['result'] == 'Away Win').mean() * 100), 1),
            'goals_by_season': matches.groupby('season')['total_goals'].mean().round(2).to_dict()
        }

        # Injury risk overview
        injury = self.gold['fact_injury_risk']
        eda['injury_overview'] = {
            'high_risk_count': int(injury['injury_risk'].sum()),
            'low_risk_count': int((injury['injury_risk'] == 0).sum()),
            'high_risk_pct': round(float(injury['injury_risk'].mean() * 100), 1),
            'avg_fatigue': round(float(injury['fatigue_index'].mean()), 2),
            'avg_training_load': round(float(injury['training_load'].mean()), 1)
        }

        return eda

    def run_full_pipeline(self):
        """Run the complete ETL pipeline"""
        print("=" * 50)
        print("Starting ETL Pipeline...")
        print("=" * 50)
        self.extract_bronze()
        self.transform_silver()
        self.load_gold()
        self.compute_data_quality()
        print("=" * 50)
        print("ETL Pipeline Complete!")
        print("=" * 50)
        return self
