"""
Fatigue Monitor — Track and analyze player fatigue levels
"""
import pandas as pd
import numpy as np


class FatigueMonitor:
    def __init__(self, etl):
        self.etl = etl

    def get_player_fatigue(self, player_api_id):
        """Get detailed fatigue analysis for a player"""
        injury = self.etl.gold['fact_injury_risk']
        perf = self.etl.gold['fact_player_performance']
        row = injury[injury['player_api_id'] == player_api_id]
        player = perf[perf['player_api_id'] == player_api_id]

        if row.empty or player.empty:
            return None

        d = row.iloc[0]
        p = player.iloc[0]
        fatigue = float(d['fatigue_index'])
        stamina = float(p['stamina']) if pd.notna(p.get('stamina')) else 60

        # Fatigue classification
        if fatigue > 7:
            level = 'Critical'
            recommendation = 'Mandatory rest. Skip next match and reduce training to recovery-only sessions.'
        elif fatigue > 5.5:
            level = 'High'
            recommendation = 'Reduce training intensity by 40%. Consider rotation for upcoming match.'
        elif fatigue > 4:
            level = 'Moderate'
            recommendation = 'Monitor closely. Maintain current load but add extra recovery protocols.'
        else:
            level = 'Low'
            recommendation = 'Player is fresh. No load management needed.'

        return {
            'player_name': str(p['player_name']),
            'fatigue_index': round(fatigue, 2),
            'fatigue_level': level,
            'stamina': round(stamina, 1),
            'wearable_metrics': {
                'avg_heart_rate': round(float(d['avg_heart_rate']), 1),
                'max_heart_rate': round(float(d['max_heart_rate']), 1),
                'sprint_distance_km': round(float(d['sprint_distance_km']), 2),
                'total_distance_km': round(float(d['total_distance_km']), 2),
                'training_load': round(float(d['training_load']), 1),
                'sleep_hours': round(float(d['sleep_hours']), 1),
                'recovery_time_hours': round(float(d['recovery_time_hours']), 1),
                'matches_last_30d': int(d['matches_last_30d']),
            },
            'recommendation': recommendation,
            'recovery_plan': self._generate_recovery_plan(level, d),
        }

    def get_squad_fatigue_report(self, top_n=20):
        """Get fatigue report for entire squad"""
        injury = self.etl.gold['fact_injury_risk']
        perf = self.etl.gold['fact_player_performance']

        merged = injury.merge(perf[['player_api_id', 'player_name', 'overall_rating', 'stamina']],
                              on='player_api_id', how='left')
        merged = merged.sort_values('fatigue_index', ascending=False).head(top_n)

        players = []
        for _, row in merged.iterrows():
            fatigue = float(row['fatigue_index'])
            level = 'Critical' if fatigue > 7 else 'High' if fatigue > 5.5 else 'Moderate' if fatigue > 4 else 'Low'
            players.append({
                'player_name': str(row['player_name']) if pd.notna(row.get('player_name')) else 'Unknown',
                'fatigue_index': round(fatigue, 2),
                'fatigue_level': level,
                'training_load': round(float(row['training_load']), 1),
                'sleep_hours': round(float(row['sleep_hours']), 1),
                'matches_last_30d': int(row['matches_last_30d']),
            })

        # Summary
        all_fatigue = injury['fatigue_index']
        summary = {
            'avg_fatigue': round(float(all_fatigue.mean()), 2),
            'critical_count': int((all_fatigue > 7).sum()),
            'high_count': int(((all_fatigue > 5.5) & (all_fatigue <= 7)).sum()),
            'moderate_count': int(((all_fatigue > 4) & (all_fatigue <= 5.5)).sum()),
            'low_count': int((all_fatigue <= 4).sum()),
        }

        return {'players': players, 'summary': summary}

    def _generate_recovery_plan(self, level, data):
        plan = []
        if level in ('Critical', 'High'):
            plan.append('🛌 **Rest**: Minimum 48 hours complete rest from training')
            plan.append('🧊 **Cold therapy**: Ice bath sessions (10 min at 10°C)')
            plan.append('💤 **Sleep optimization**: Target 9+ hours, blackout room, no screens 1hr before bed')
            if float(data['sleep_hours']) < 7:
                plan.append('⚠️ **Sleep deficit detected**: Current avg is only {:.1f}hrs. Critical to address.'.format(float(data['sleep_hours'])))
        if level in ('Critical', 'High', 'Moderate'):
            plan.append('🏊 **Active recovery**: Light swimming or cycling only (HR < 120 bpm)')
            plan.append('🧘 **Mobility work**: Daily stretching and yoga sessions (30 min)')
            plan.append('🩺 **Medical check**: Blood markers (CK, cortisol) and muscle screening')
        if int(data['matches_last_30d']) > 5:
            plan.append('📅 **Schedule management**: Excessive match load ({} in 30 days). Must skip at least 1 upcoming fixture.'.format(int(data['matches_last_30d'])))
        if level == 'Low':
            plan.append('✅ **No action needed**: Continue normal training program')
            plan.append('📊 **Maintain monitoring**: Regular check-ins every 72 hours')
        return plan


class InjuryPredictor:
    """Enhanced injury prediction with explainable factors"""
    def __init__(self, etl, ml):
        self.etl = etl
        self.ml = ml

    def predict_with_explanation(self, player_api_id):
        """Predict injury risk with full SHAP-like explanation"""
        risk = self.ml.predict_injury_risk(player_api_id)
        if not risk:
            return None

        perf = self.etl.gold['fact_player_performance']
        player = perf[perf['player_api_id'] == player_api_id]
        if player.empty:
            return risk

        p = player.iloc[0]
        wd = risk.get('wearable_data', {})

        # Feature contribution analysis
        contributions = []
        fatigue = wd.get('fatigue_index', 5)
        if fatigue > 6:
            contributions.append({'factor': 'Fatigue Index', 'value': fatigue, 'impact': 'HIGH',
                                  'explanation': f'Fatigue index of {fatigue:.1f} exceeds safe threshold (6.0)'})
        elif fatigue > 4.5:
            contributions.append({'factor': 'Fatigue Index', 'value': fatigue, 'impact': 'MODERATE',
                                  'explanation': f'Fatigue index of {fatigue:.1f} is elevated but manageable'})

        load = wd.get('training_load', 500)
        if load > 600:
            contributions.append({'factor': 'Training Load', 'value': load, 'impact': 'HIGH',
                                  'explanation': f'Training load of {load:.0f} indicates overtraining risk'})

        sleep = wd.get('sleep_hours', 7)
        if sleep < 6.5:
            contributions.append({'factor': 'Sleep Quality', 'value': sleep, 'impact': 'HIGH',
                                  'explanation': f'Only {sleep:.1f} hours of sleep — insufficient recovery'})

        injuries = wd.get('previous_injuries', 0)
        if injuries > 3:
            contributions.append({'factor': 'Injury History', 'value': injuries, 'impact': 'HIGH',
                                  'explanation': f'{injuries} previous injuries — significantly higher re-injury probability'})
        elif injuries > 1:
            contributions.append({'factor': 'Injury History', 'value': injuries, 'impact': 'MODERATE',
                                  'explanation': f'{injuries} previous injuries — monitor vulnerable areas'})

        matches = wd.get('matches_last_30d', 4)
        if matches > 5:
            contributions.append({'factor': 'Match Congestion', 'value': matches, 'impact': 'HIGH',
                                  'explanation': f'{matches} matches in 30 days — excessive fixture load'})

        recovery = wd.get('recovery_time_hours', 36)
        if recovery > 45:
            contributions.append({'factor': 'Recovery Time', 'value': recovery, 'impact': 'MODERATE',
                                  'explanation': f'Recovery time of {recovery:.0f}h suggests slow physical recovery'})

        age = float(p['age']) if pd.notna(p['age']) else 25
        if age > 32:
            contributions.append({'factor': 'Age', 'value': round(age, 0), 'impact': 'MODERATE',
                                  'explanation': f'Age {age:.0f} increases baseline injury susceptibility'})

        risk['contributions'] = contributions
        risk['player_name'] = str(p['player_name'])
        return risk

    def get_high_risk_squad(self, top_n=15):
        """Get players with highest injury risk"""
        injury = self.etl.gold['fact_injury_risk']
        perf = self.etl.gold['fact_player_performance']

        high_risk = injury[injury['injury_risk'] == 1].merge(
            perf[['player_api_id', 'player_name', 'overall_rating', 'age']],
            on='player_api_id', how='left'
        ).sort_values('fatigue_index', ascending=False).head(top_n)

        result = []
        for _, row in high_risk.iterrows():
            result.append({
                'player_name': str(row['player_name']) if pd.notna(row.get('player_name')) else 'Unknown',
                'overall_rating': round(float(row['overall_rating']), 0) if pd.notna(row.get('overall_rating')) else None,
                'age': round(float(row['age']), 0) if pd.notna(row.get('age')) else None,
                'fatigue_index': round(float(row['fatigue_index']), 2),
                'training_load': round(float(row['training_load']), 1),
                'previous_injuries': int(row['previous_injuries']),
            })

        return result
