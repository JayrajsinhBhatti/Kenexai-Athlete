"""
Anomaly Detector — Detect anomalies in player performance and fitness metrics
Uses Isolation Forest and Z-score methods
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats


class AnomalyDetector:
    def __init__(self, etl):
        self.etl = etl
        self.iso_model = None

    def detect_all_anomalies(self, top_n=15):
        """Run full anomaly detection pipeline"""
        anomalies = []

        # 1. Performance anomalies (Z-score)
        perf_anomalies = self._detect_performance_anomalies()
        anomalies.extend(perf_anomalies)

        # 2. Fatigue anomalies (Isolation Forest)
        fatigue_anomalies = self._detect_fatigue_anomalies()
        anomalies.extend(fatigue_anomalies)

        # 3. Workload anomalies
        workload_anomalies = self._detect_workload_anomalies()
        anomalies.extend(workload_anomalies)

        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MODERATE': 2}
        anomalies.sort(key=lambda x: severity_order.get(x.get('severity', 'MODERATE'), 3))

        return anomalies[:top_n]

    def _detect_performance_anomalies(self):
        """Detect statistical outliers in performance using Z-scores"""
        perf = self.etl.gold['fact_player_performance']
        anomalies = []

        check_cols = ['overall_rating', 'performance_score', 'attack_score',
                      'midfield_score', 'defense_score', 'physical_score']

        for col in check_cols:
            if col not in perf.columns:
                continue
            values = perf[col].dropna()
            if len(values) < 10:
                continue

            z_scores = np.abs(stats.zscore(values))
            outlier_mask = z_scores > 2.5
            outlier_indices = values.index[outlier_mask]

            for idx in outlier_indices[:3]:
                player = perf.loc[idx]
                z = float(z_scores[values.index.get_loc(idx)])
                val = float(values.loc[idx])
                mean = float(values.mean())

                direction = 'above' if val > mean else 'below'
                severity = 'CRITICAL' if z > 3 else 'HIGH' if z > 2.5 else 'MODERATE'

                anomalies.append({
                    'type': 'performance_outlier',
                    'player_name': str(player['player_name']),
                    'player_api_id': int(player['player_api_id']),
                    'metric': col.replace('_', ' ').title(),
                    'value': round(val, 1),
                    'league_average': round(mean, 1),
                    'z_score': round(z, 2),
                    'direction': direction,
                    'severity': severity,
                    'explanation': f"{player['player_name']}'s {col.replace('_', ' ')} ({val:.1f}) "
                                  f"is {z:.1f} std devs {direction} avg ({mean:.1f})"
                })

        return anomalies

    def _detect_fatigue_anomalies(self):
        """Detect abnormal fatigue patterns using Isolation Forest"""
        injury = self.etl.gold['fact_injury_risk']
        perf = self.etl.gold['fact_player_performance']
        anomalies = []

        features = ['fatigue_index', 'training_load', 'sleep_hours',
                     'matches_last_30d', 'recovery_time_hours']
        X = injury[features].fillna(0)

        self.iso_model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        predictions = self.iso_model.fit_predict(X)
        scores = self.iso_model.score_samples(X)

        anomaly_mask = predictions == -1
        anomaly_indices = injury.index[anomaly_mask]

        for idx in anomaly_indices[:10]:
            row = injury.loc[idx]
            score = float(scores[injury.index.get_loc(idx)])
            pname = None

            player = perf[perf['player_api_id'] == row['player_api_id']]
            if not player.empty:
                pname = str(player.iloc[0]['player_name'])

            reasons = []
            if float(row['fatigue_index']) > 7:
                reasons.append(f"Fatigue index critically high ({row['fatigue_index']:.1f})")
            if float(row['training_load']) > 650:
                reasons.append(f"Extreme training load ({row['training_load']:.0f})")
            if float(row['sleep_hours']) < 5.5:
                reasons.append(f"Severe sleep deficit ({row['sleep_hours']:.1f}h)")
            if int(row['matches_last_30d']) > 6:
                reasons.append(f"Excessive matches ({int(row['matches_last_30d'])} in 30 days)")
            if float(row['recovery_time_hours']) > 50:
                reasons.append(f"Prolonged recovery time ({row['recovery_time_hours']:.0f}h)")

            if not reasons:
                reasons.append("Unusual combination of fitness metrics")

            anomalies.append({
                'type': 'fatigue_anomaly',
                'player_name': pname or f"Player {row['player_api_id']}",
                'player_api_id': int(row['player_api_id']),
                'anomaly_score': round(abs(score), 3),
                'severity': 'CRITICAL' if abs(score) > 0.3 else 'HIGH',
                'fatigue_index': round(float(row['fatigue_index']), 2),
                'training_load': round(float(row['training_load']), 1),
                'reasons': reasons,
                'explanation': f"{pname or 'Player'}: {'; '.join(reasons)}"
            })

        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
        return anomalies

    def _detect_workload_anomalies(self):
        """Detect abnormal workload patterns via Z-score"""
        injury = self.etl.gold['fact_injury_risk']
        perf = self.etl.gold['fact_player_performance']
        anomalies = []

        load = injury['training_load']
        mean, std = float(load.mean()), float(load.std())

        high_load = injury[load > mean + 2 * std].head(5)
        for _, row in high_load.iterrows():
            player = perf[perf['player_api_id'] == row['player_api_id']]
            pname = str(player.iloc[0]['player_name']) if not player.empty else f"Player {row['player_api_id']}"
            anomalies.append({
                'type': 'workload_anomaly',
                'player_name': pname,
                'player_api_id': int(row['player_api_id']),
                'training_load': round(float(row['training_load']), 1),
                'league_avg_load': round(mean, 1),
                'severity': 'HIGH',
                'explanation': f"{pname}'s training load ({row['training_load']:.0f}) is {((row['training_load'] - mean) / std):.1f}x std above average ({mean:.0f})"
            })

        return anomalies
