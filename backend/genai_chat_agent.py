"""
GenAI Chat Agent — Advanced Sports Analyst Assistant
Professional-grade NLP with intent detection, data retrieval, ML execution,
and sports analyst-style insight generation.
"""
import re
import random
import pandas as pd
import numpy as np


ANALYST_INTENTS = [
    # Player analysis
    {'intent': 'player_profile', 'patterns': [
        r'(?:tell me about|profile|info|details|who is|show me) (.+)',
        r'(?:analyse|analyze|analysis of|analyze|assessment of) (.+)',
    ]},
    # Injury risk
    {'intent': 'injury_risk', 'patterns': [
        r'(?:injury|injured|injury risk|injury prediction|risk of injury)(?: for| of)? (.+)',
        r'(?:will|is|could) (.+?) (?:get injured|be injured|injury)',
        r'(?:explain|why) (.+?) (?:has |is )?(?:high|injury) risk',
    ]},
    # Fatigue
    {'intent': 'fatigue', 'patterns': [
        r'(?:fatigue|tired|exhausted|fatigue trends?|fatigue level)(?: for| of)? (.+)',
        r'(?:show|get) fatigue (?:for|of|trends?) (.+)',
        r'(.+?) fatigue',
    ]},
    # Top players / rankings
    {'intent': 'top_players', 'patterns': [
        r'(?:top|best|highest|leading) (\d+)? ?(?:players?|performers?)(?: (?:by|in|with|for) (.+))?',
        r'(?:who are|show me) (?:the )?(?:top|best) (.+)',
        r'(?:player|performance) (?:rankings?|leaderboard)',
    ]},
    # Declining players
    {'intent': 'declining_players', 'patterns': [
        r'(?:declining|decreasing|dropping|falling|worsening) (?:performance|players?|rating)',
        r'(?:players?|who) (?:are|show|with) (?:declining|dropping|decreasing)',
        r'performance (?:decline|drop)',
    ]},
    # Improving players
    {'intent': 'improving_players', 'patterns': [
        r'(?:improving|rising|growing|developing|promising) (?:players?|performance)',
        r'(?:players?|who) (?:are|show|with) (?:improving|rising|getting better)',
    ]},
    # Comparison
    {'intent': 'compare', 'patterns': [
        r'compare (.+?) (?:vs?\.?|versus|against|and|with) (.+)',
        r'(.+?) (?:vs?\.?|versus|against) (.+)',
    ]},
    # Lineup / team recommendation
    {'intent': 'lineup', 'patterns': [
        r'(?:recommend|suggest|optimal|best|starting) (?:lineup|xi|eleven|team|formation)(?: for (.+))?',
        r'(?:lineup|starting|formation) (?:for|recommendation) (.+)',
    ]},
    # Match prediction
    {'intent': 'match_prediction', 'patterns': [
        r'(?:predict|prediction|forecast) (.+?) (?:vs?\.?|versus|against) (.+)',
        r'(.+?) (?:vs?\.?|versus|against) (.+?) (?:prediction|result|outcome)',
        r'(?:who will win|who wins) (.+?) (?:vs?\.?|or|against) (.+)',
    ]},
    # Anomalies
    {'intent': 'anomalies', 'patterns': [
        r'(?:anomal|unusual|abnormal|irregular|strange|weird)',
        r'(?:detect|find|show) (?:anomal|unusual|outlier)',
    ]},
    # Team info
    {'intent': 'team_info', 'patterns': [
        r'(?:tell me about|info|details|analysis)(?: team| on)? team (.+)',
        r'(.+?) (?:team|squad|roster) (?:info|details|analysis)',
    ]},
    # Stats / general
    {'intent': 'statistics', 'patterns': [
        r'(?:overall |general )?(?:stats?|statistics|numbers|overview|summary|dashboard)',
        r'(?:how many|count|total) (?:players?|matches?|teams?|goals?)',
    ]},
    # Head-to-head
    {'intent': 'head_to_head', 'patterns': [
        r'(?:head to head|h2h|history|record) (.+?) (?:vs?\.?|versus|against|and) (.+)',
        r'(?:matches?|games?) (?:between|of) (.+?) (?:and|vs?\.?) (.+)',
    ]},
    # Coaching advice
    {'intent': 'coaching', 'patterns': [
        r'(?:coaching|training|development|improve|improvement) (?:advice|plan|recommendation|tips?)(?: for (.+))?',
        r'(?:how (?:to|can)|ways to) (?:improve|develop|train) (.+)',
    ]},
]


class GenAIChatAgent:
    def __init__(self, etl, ml, data_engine, perf_analyzer, fatigue_monitor,
                 injury_predictor, anomaly_detector, lineup_optimizer, chart_generator):
        self.etl = etl
        self.ml = ml
        self.data_engine = data_engine
        self.perf_analyzer = perf_analyzer
        self.fatigue_monitor = fatigue_monitor
        self.injury_predictor = injury_predictor
        self.anomaly_detector = anomaly_detector
        self.lineup_optimizer = lineup_optimizer
        self.chart_generator = chart_generator
        self.conversation_history = []

    def process_message(self, message):
        """Main entry point — process user message and generate analyst response"""
        self.conversation_history.append({'role': 'user', 'message': message})

        intent, entities = self._detect_intent(message)
        response = self._dispatch(intent, entities, message)

        self.conversation_history.append({'role': 'assistant', 'response': response})
        return response

    def _detect_intent(self, message):
        msg = message.lower().strip()

        for intent_def in ANALYST_INTENTS:
            for pattern in intent_def['patterns']:
                match = re.search(pattern, msg, re.IGNORECASE)
                if match:
                    return intent_def['intent'], match.groups()

        return 'general', ()

    def _dispatch(self, intent, entities, message):
        handlers = {
            'player_profile': self._handle_player_profile,
            'injury_risk': self._handle_injury_risk,
            'fatigue': self._handle_fatigue,
            'top_players': self._handle_top_players,
            'declining_players': self._handle_declining,
            'improving_players': self._handle_improving,
            'compare': self._handle_compare,
            'lineup': self._handle_lineup,
            'match_prediction': self._handle_match_prediction,
            'anomalies': self._handle_anomalies,
            'team_info': self._handle_team_info,
            'statistics': self._handle_statistics,
            'head_to_head': self._handle_head_to_head,
            'coaching': self._handle_coaching,
        }

        handler = handlers.get(intent, self._handle_general)
        try:
            return handler(entities, message)
        except Exception as e:
            return self._error_response(str(e))

    # ─────────── HANDLERS ───────────

    def _handle_player_profile(self, entities, message):
        name = self._extract_player_name(entities, message)
        player_id = self._find_player_id(name)
        if not player_id:
            return self._player_not_found(name)

        analysis = self.perf_analyzer.analyze_player(player_id)
        chart = self.chart_generator.player_performance_radar(player_id)
        trend_chart = self.chart_generator.performance_trend(player_id)
        injury = self.injury_predictor.predict_with_explanation(player_id)
        fatigue = self.fatigue_monitor.get_player_fatigue(player_id)

        p = analysis['player']
        msg = f"## 📊 Player Analysis: **{p['name']}**\n\n"
        msg += f"**Age:** {p['age']:.0f} | **Overall:** {p['overall']} | **Potential:** {p['potential']} | **Performance Score:** {p['performance_score']}\n\n"

        msg += "### Skill Assessment\n"
        msg += f"- ⚔️ **Attack:** {analysis['scores']['attack']:.0f}/100\n"
        msg += f"- 🎯 **Midfield:** {analysis['scores']['midfield']:.0f}/100\n"
        msg += f"- 🛡️ **Defense:** {analysis['scores']['defense']:.0f}/100\n"
        msg += f"- 💪 **Physical:** {analysis['scores']['physical']:.0f}/100\n\n"

        msg += f"**Best Position:** {analysis['position_fit']['best_position']} ({analysis['position_fit']['versatility']})\n"
        msg += f"**Development Gap:** {analysis['development_gap']:.0f} (Potential - Current)\n\n"

        if analysis['trend']:
            trend = analysis['trend']
            icon = '📈' if trend['direction'] == 'improving' else '📉' if trend['direction'] == 'declining' else '➡️'
            msg += f"### Performance Trend\n"
            msg += f"{icon} **{trend['direction'].title()}** ({trend['change']:+.1f}) over {trend['data_points']} observations\n\n"

        msg += "### Top Strengths\n"
        for s in analysis['strengths'][:3]:
            msg += f"- ✅ **{s['skill']}:** {s['value']:.0f}\n"

        msg += "\n### Areas for Improvement\n"
        for w in analysis['weaknesses'][:3]:
            msg += f"- 📈 **{w['skill']}:** {w['value']:.0f}\n"

        if injury:
            msg += f"\n### Injury Risk Assessment\n"
            risk_icon = '🔴' if injury['risk_level'] == 'High' else '🟢'
            msg += f"{risk_icon} **Risk Level:** {injury['risk_level']} ({injury['risk_probability']}%)\n"
            for f in injury.get('risk_factors', [])[:3]:
                msg += f"- ⚠️ {f}\n"

        response = {
            'type': 'player_analysis',
            'message': msg,
            'player': analysis,
            'charts': [chart, trend_chart],
            'injury_risk': injury,
            'fatigue': fatigue,
            'suggestions': [
                f"Compare {p['name']} with another player",
                f"Injury risk for {p['name']}",
                f"Coaching plan for {p['name']}",
                f"Show fatigue trends for {p['name']}"
            ]
        }
        return response

    def _handle_injury_risk(self, entities, message):
        name = self._extract_player_name(entities, message)
        player_id = self._find_player_id(name)
        if not player_id:
            return self._player_not_found(name)

        risk = self.injury_predictor.predict_with_explanation(player_id)
        fatigue = self.fatigue_monitor.get_player_fatigue(player_id)

        if not risk:
            return self._player_not_found(name)

        pname = risk.get('player_name', name)
        risk_icon = '🔴' if risk['risk_level'] == 'High' else '🟢'

        msg = f"## 🏥 Injury Risk Analysis: **{pname}**\n\n"
        msg += f"{risk_icon} **Overall Risk Level:** {risk['risk_level']} ({risk['risk_probability']}%)\n\n"

        if risk.get('contributions'):
            msg += "### Contributing Factors\n\n"
            msg += "| Factor | Value | Impact | Analysis |\n"
            msg += "|--------|-------|--------|----------|\n"
            for c in risk['contributions']:
                impact_icon = '🔴' if c['impact'] == 'HIGH' else '🟡' if c['impact'] == 'MODERATE' else '🟢'
                msg += f"| {c['factor']} | {c['value']:.1f} | {impact_icon} {c['impact']} | {c['explanation']} |\n"
            msg += "\n"

        if risk.get('wearable_data'):
            wd = risk['wearable_data']
            msg += "### Wearable Sensor Data\n"
            msg += f"- ❤️ Avg Heart Rate: {wd['avg_heart_rate']:.0f} bpm | Max: {wd['max_heart_rate']:.0f} bpm\n"
            msg += f"- 🏃 Sprint Distance: {wd['sprint_distance_km']:.1f} km\n"
            msg += f"- 📊 Training Load: {wd['training_load']:.0f}\n"
            msg += f"- 😴 Sleep: {wd['sleep_hours']:.1f} hours\n"
            msg += f"- 🔋 Fatigue Index: {wd['fatigue_index']:.1f}/10\n\n"

        if fatigue:
            msg += f"### 🩺 Medical Recommendation\n"
            msg += f"{fatigue['recommendation']}\n\n"
            if fatigue.get('recovery_plan'):
                msg += "### Recovery Protocol\n"
                for step in fatigue['recovery_plan']:
                    msg += f"{step}\n"

        return {
            'type': 'injury_analysis',
            'message': msg,
            'risk_data': risk,
            'fatigue_data': fatigue,
            'suggestions': [
                f"Fatigue trends for {pname}",
                f"Coaching plan for {pname}",
                "Show all high-risk players",
                "Detect anomalies"
            ]
        }

    def _handle_fatigue(self, entities, message):
        name = self._extract_player_name(entities, message)
        player_id = self._find_player_id(name)

        if player_id:
            fatigue = self.fatigue_monitor.get_player_fatigue(player_id)
            if not fatigue:
                return self._player_not_found(name)

            level_icon = {'Critical': '🔴', 'High': '🟠', 'Moderate': '🟡', 'Low': '🟢'}.get(fatigue['fatigue_level'], '⚪')

            msg = f"## 🔋 Fatigue Report: **{fatigue['player_name']}**\n\n"
            msg += f"{level_icon} **Fatigue Level:** {fatigue['fatigue_level']} (Index: {fatigue['fatigue_index']:.1f}/10)\n\n"
            msg += "### Biometric Data\n"
            wm = fatigue['wearable_metrics']
            msg += f"- ❤️ Heart Rate: {wm['avg_heart_rate']:.0f} bpm (Max: {wm['max_heart_rate']:.0f})\n"
            msg += f"- 🏃 Sprint Distance: {wm['sprint_distance_km']:.1f} km | Total: {wm['total_distance_km']:.1f} km\n"
            msg += f"- 📊 Training Load: {wm['training_load']:.0f}\n"
            msg += f"- 😴 Sleep: {wm['sleep_hours']:.1f}h\n"
            msg += f"- 📅 Matches (30 days): {wm['matches_last_30d']}\n\n"
            msg += f"### 📋 Recommendation\n{fatigue['recommendation']}\n\n"

            if fatigue.get('recovery_plan'):
                msg += "### Recovery Protocol\n"
                for step in fatigue['recovery_plan']:
                    msg += f"{step}\n"

            return {
                'type': 'fatigue_report',
                'message': msg,
                'fatigue_data': fatigue,
                'suggestions': [f"Injury risk for {fatigue['player_name']}", "Squad fatigue overview", "Detect anomalies"]
            }
        else:
            # Squad-level fatigue
            report = self.fatigue_monitor.get_squad_fatigue_report()
            chart = self.chart_generator.fatigue_distribution()

            msg = "## 🔋 Squad Fatigue Overview\n\n"
            s = report['summary']
            msg += f"| Level | Count |\n|-------|-------|\n"
            msg += f"| 🔴 Critical | {s['critical_count']} |\n"
            msg += f"| 🟠 High | {s['high_count']} |\n"
            msg += f"| 🟡 Moderate | {s['moderate_count']} |\n"
            msg += f"| 🟢 Low | {s['low_count']} |\n\n"
            msg += f"**Average Fatigue:** {s['avg_fatigue']:.1f}/10\n\n"

            if report['players']:
                msg += "### Most Fatigued Players\n"
                for p in report['players'][:5]:
                    icon = '🔴' if p['fatigue_level'] == 'Critical' else '🟠' if p['fatigue_level'] == 'High' else '🟡'
                    msg += f"- {icon} **{p['player_name']}**: Fatigue {p['fatigue_index']:.1f}, Load {p['training_load']:.0f}\n"

            return {
                'type': 'squad_fatigue',
                'message': msg,
                'report': report,
                'charts': [chart],
                'suggestions': ["Show high-risk players", "Detect anomalies", "Recommend lineup"]
            }

    def _handle_top_players(self, entities, message):
        n = 10
        metric = 'performance_score'
        for e in entities:
            if e and e.isdigit():
                n = min(int(e), 25)
            elif e:
                attr_map = {
                    'attack': 'attack_score', 'defense': 'defense_score', 'midfield': 'midfield_score',
                    'physical': 'physical_score', 'rating': 'overall_rating', 'overall': 'overall_rating',
                    'potential': 'potential', 'performance': 'performance_score', 'speed': 'sprint_speed',
                    'stamina': 'stamina', 'finishing': 'finishing', 'dribbling': 'dribbling',
                }
                metric = attr_map.get(e.strip().lower(), metric)

        perf = self.etl.gold['fact_player_performance']
        top = perf.nlargest(n, metric)
        chart = self.chart_generator.top_players_chart(metric, n)

        msg = f"## 🏆 Top {n} Players by {metric.replace('_', ' ').title()}\n\n"
        msg += "| Rank | Player | Age | Rating | Performance | Score |\n"
        msg += "|------|--------|-----|--------|-------------|-------|\n"
        for i, (_, p) in enumerate(top.iterrows()):
            medal = ['🥇', '🥈', '🥉'][i] if i < 3 else f"#{i+1}"
            msg += f"| {medal} | **{p['player_name']}** | {p['age']:.0f} | {p['overall_rating']:.0f} | {p['performance_score']:.1f} | {p.get(metric, 0):.1f} |\n"
        msg += "\n"

        msg += f"*Ranked by {metric.replace('_', ' ').title()}. League average: {perf[metric].mean():.1f}*"

        return {
            'type': 'leaderboard',
            'message': msg,
            'players': self._df_to_list(top),
            'charts': [chart],
            'metric': metric,
            'suggestions': ["Show declining players", "Show improving players", "Compare top 2 players", "Recommend lineup"]
        }

    def _handle_declining(self, entities, message):
        declining = self.perf_analyzer.find_declining_players()

        msg = "## 📉 Players with Declining Performance\n\n"
        msg += "These players show significant rating drops based on historical data.\n\n"
        msg += "| Player | Current | Previous | Decline | Observations |\n"
        msg += "|--------|---------|----------|---------|-------------|\n"
        for p in declining[:10]:
            msg += f"| **{p['player_name']}** | {p['current_rating']:.0f} | {p['previous_rating']:.0f} | 🔻 -{p['decline']:.1f} | {p['data_points']} |\n"

        msg += "\n> ⚠️ **Analyst Note:** Players showing consistent decline may benefit from adjusted training programs, position changes, or tactical role modifications.\n"

        return {
            'type': 'declining_report',
            'message': msg,
            'players': declining[:15],
            'suggestions': ["Show improving players", "Coaching advice for top decliner", "Detect anomalies"]
        }

    def _handle_improving(self, entities, message):
        improving = self.perf_analyzer.find_improving_players()

        msg = "## 📈 Players with Improving Performance\n\n"
        msg += "| Player | Current | Previous | Improvement |\n"
        msg += "|--------|---------|----------|-------------|\n"
        for p in improving[:10]:
            msg += f"| **{p['player_name']}** | {p['current_rating']:.0f} | {p['previous_rating']:.0f} | 🔺 +{p['improvement']:.1f} |\n"

        msg += "\n> 💡 **Analyst Note:** These players represent excellent development trajectories and potential investment opportunities.\n"

        return {
            'type': 'improving_report',
            'message': msg,
            'players': improving[:15],
            'suggestions': ["Show declining players", "Top 10 players", "Recommend lineup"]
        }

    def _handle_compare(self, entities, message):
        if len(entities) < 2:
            return self._error_response("Please specify two players to compare (e.g., 'Compare Messi vs Ronaldo')")

        name1 = entities[0].strip().rstrip('?.,! ')
        name2 = entities[1].strip().rstrip('?.,! ')
        id1 = self._find_player_id(name1)
        id2 = self._find_player_id(name2)

        if not id1:
            return self._player_not_found(name1)
        if not id2:
            return self._player_not_found(name2)

        comp = self.perf_analyzer.compare_players(id1, id2)
        chart = self.chart_generator.player_comparison_chart(id1, id2)

        if not comp:
            return self._error_response("Could not retrieve comparison data")

        p1, p2 = comp['player1']['player'], comp['player2']['player']

        msg = f"## ⚔️ Player Comparison: **{p1['name']}** vs **{p2['name']}**\n\n"
        msg += f"| Attribute | {p1['name']} | {p2['name']} | Edge |\n"
        msg += "|-----------|-------------|-------------|------|\n"

        for attr, vals in comp['overall_comparison'].items():
            v1 = vals['player1'] or 0
            v2 = vals['player2'] or 0
            edge = '⬅️' if v1 > v2 else '➡️' if v2 > v1 else '🤝'
            msg += f"| **{attr.replace('_', ' ').title()}** | {v1} | {v2} | {edge} {vals['winner']} |\n"

        for attr, vals in comp['skill_comparison'].items():
            v1, v2 = vals['player1'], vals['player2']
            edge = '⬅️' if v1 > v2 else '➡️' if v2 > v1 else '🤝'
            msg += f"| **{attr.title()}** | {v1:.0f} | {v2:.0f} | {edge} {vals['winner']} |\n"

        msg += "\n### 🧠 Analyst Verdict\n"
        p1_wins = sum(1 for v in comp['skill_comparison'].values() if v['player1'] > v['player2'])
        p2_wins = sum(1 for v in comp['skill_comparison'].values() if v['player2'] > v['player1'])
        if p1_wins > p2_wins:
            msg += f"**{p1['name']}** edges ahead in {p1_wins} out of {len(comp['skill_comparison'])} skill categories. "
        elif p2_wins > p1_wins:
            msg += f"**{p2['name']}** edges ahead in {p2_wins} out of {len(comp['skill_comparison'])} skill categories. "
        else:
            msg += f"Both players are evenly matched across skill categories. "

        msg += f"{p1['name']} is better suited as a {comp['player1']['position_fit']['best_position']}, "
        msg += f"while {p2['name']} fits best as a {comp['player2']['position_fit']['best_position']}.\n"

        return {
            'type': 'comparison',
            'message': msg,
            'comparison': comp,
            'charts': [chart],
            'suggestions': [
                f"Injury risk for {p1['name']}",
                f"Injury risk for {p2['name']}",
                f"Coaching plan for {p1['name']}"
            ]
        }

    def _handle_lineup(self, entities, message):
        team_name = None
        formation = '4-3-3'

        for e in entities:
            if e:
                e = e.strip().rstrip('?.,!')
                if e in FORMATIONS if hasattr(self, '_') else e in ['4-3-3', '4-4-2', '3-5-2', '4-2-3-1', '3-4-3']:
                    formation = e
                else:
                    team_name = e

        # Extract formation from message
        for f in ['4-3-3', '4-4-2', '3-5-2', '4-2-3-1', '3-4-3']:
            if f in message:
                formation = f

        result = self.lineup_optimizer.recommend_lineup(team_name=team_name, formation=formation)

        if 'error' in result:
            return self._error_response(result['error'])

        msg = f"## ⚽ Recommended Starting XI"
        if team_name:
            msg += f" — {team_name.title()}"
        msg += f"\n**Formation: {formation}**\n\n"

        pos_groups = {}
        for p in result['lineup']:
            pos_groups.setdefault(p['position'], []).append(p)

        pos_order = ['GK', 'DEF', 'MID', 'FWD']
        pos_labels = {'GK': '🧤 GOALKEEPER', 'DEF': '🛡️ DEFENDERS', 'MID': '🎯 MIDFIELD', 'FWD': '⚔️ FORWARDS'}

        for pos in pos_order:
            if pos in pos_groups:
                msg += f"\n**{pos_labels.get(pos, pos)}**\n"
                for p in pos_groups[pos]:
                    msg += f"- {p['fitness_status']} **{p['player_name']}** (Rating: {p['overall_rating']}, "
                    msg += f"Performance: {p['performance_score']}, Fatigue: {p['fatigue_index']:.1f})\n"

        ts = result['team_stats']
        msg += f"\n### 📊 Team Statistics\n"
        msg += f"- Avg Rating: **{ts['avg_rating']}** | Avg Performance: **{ts['avg_performance']}**\n"
        msg += f"- Squad Fitness: **{ts['squad_fitness']}** (Avg Fatigue: {ts['avg_fatigue']:.1f})\n"
        msg += f"- Players at Injury Risk: **{ts['high_risk_count']}**\n"
        msg += f"- Team Chemistry: **{result['chemistry']['level']}** ({result['chemistry']['score']}%)\n"

        if result.get('bench'):
            msg += f"\n### 🪑 Bench\n"
            for p in result['bench'][:5]:
                msg += f"- {p['player_name']} ({p['position']}, Rating: {p['overall_rating']})\n"

        return {
            'type': 'lineup',
            'message': msg,
            'lineup_data': result,
            'suggestions': ["Predict match outcome", "Squad fatigue report", "Show anomalies"]
        }

    def _handle_match_prediction(self, entities, message):
        if len(entities) < 2:
            return self._error_response("Please specify two teams (e.g., 'Predict Barcelona vs Real Madrid')")

        team1 = entities[0].strip().rstrip('?.,! ')
        team2 = entities[1].strip().rstrip('?.,! ')

        # Find team IDs
        teams = self.etl.gold['dim_team']
        t1 = teams[teams['team_long_name'].str.lower().str.contains(team1.lower(), na=False)]
        t2 = teams[teams['team_long_name'].str.lower().str.contains(team2.lower(), na=False)]

        if t1.empty:
            return self._error_response(f"Team '{team1}' not found in database")
        if t2.empty:
            return self._error_response(f"Team '{team2}' not found in database")

        t1_id = int(t1.iloc[0]['team_api_id'])
        t2_id = int(t2.iloc[0]['team_api_id'])
        t1_name = str(t1.iloc[0]['team_long_name'])
        t2_name = str(t2.iloc[0]['team_long_name'])

        prediction = self.ml.predict_match(t1_id, t2_id)
        h2h = self.data_engine.head_to_head(
            re.search(r'(.*)', team1), message
        ) if hasattr(self.data_engine, 'head_to_head') else None

        msg = f"## ⚽ Match Prediction: **{t1_name}** vs **{t2_name}**\n\n"

        if 'error' in prediction:
            msg += f"⚠️ {prediction['error']}\n"
        else:
            msg += f"### 🔮 Predicted Outcome: **{prediction['prediction']}**\n\n"
            msg += "| Outcome | Probability |\n|---------|-------------|\n"
            for outcome, prob in prediction.get('probabilities', {}).items():
                bar = '█' * int(prob / 5) + '░' * (20 - int(prob / 5))
                msg += f"| **{outcome}** | {bar} {prob}% |\n"

            msg += "\n> 🧠 *Prediction based on team tactical attributes (build-up speed, defence pressure, chance creation) using Gradient Boosting model.*\n"

        return {
            'type': 'match_prediction',
            'message': msg,
            'prediction': prediction,
            'suggestions': [
                f"Recommend lineup for {t1_name}",
                f"Head to head {t1_name} vs {t2_name}",
                f"Team analysis {t1_name}"
            ]
        }

    def _handle_anomalies(self, entities, message):
        anomalies = self.anomaly_detector.detect_all_anomalies()

        msg = "## 🚨 Anomaly Detection Report\n\n"
        msg += f"Found **{len(anomalies)}** anomalies using Isolation Forest and Z-score analysis.\n\n"

        for i, anomaly in enumerate(anomalies[:10]):
            sev_icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MODERATE': '🟡'}.get(anomaly.get('severity'), '⚪')
            msg += f"### {sev_icon} Anomaly #{i+1}: {anomaly.get('type', 'unknown').replace('_', ' ').title()}\n"
            msg += f"**Player:** {anomaly.get('player_name', 'Unknown')} | **Severity:** {anomaly['severity']}\n"
            msg += f"📝 {anomaly.get('explanation', '')}\n\n"

        msg += "\n> 🤖 *Analysis performed using Isolation Forest (contamination=10%) for multivariate anomaly detection and Z-score (threshold=2.5σ) for univariate outlier detection.*\n"

        return {
            'type': 'anomaly_report',
            'message': msg,
            'anomalies': anomalies,
            'suggestions': ["Show high-risk players", "Squad fatigue overview", "Show declining players"]
        }

    def _handle_team_info(self, entities, message):
        name = entities[0].strip().rstrip('?.,!') if entities and entities[0] else self._extract_team_name(message)
        result = self.data_engine.players_from_team(re.search(r'(.+)', name), message) if name else None

        if not result or result.get('type') == 'error':
            return self._error_response(f"Team '{name}' not found")

        msg = f"## 🏟️ Team Analysis: **{result.get('team', name)}**\n\n"
        msg += f"**Squad Size:** {result['count']} players found\n\n"
        msg += "| # | Player | Age | Rating | Performance |\n"
        msg += "|---|--------|-----|--------|-------------|\n"
        for i, p in enumerate(result['data'][:15]):
            msg += f"| {i+1} | **{p['player_name']}** | {p.get('age', '-')} | {p.get('overall_rating', '-')} | {p.get('performance_score', '-')} |\n"

        return {
            'type': 'team_analysis',
            'message': msg,
            'team_data': result,
            'suggestions': [f"Recommend lineup for {result.get('team', name)}", f"Predict {result.get('team', name)} match"]
        }

    def _handle_statistics(self, entities, message):
        perf = self.etl.gold['fact_player_performance']
        matches = self.etl.gold['fact_match']
        teams = self.etl.gold['dim_team']
        injury = self.etl.gold['fact_injury_risk']
        chart_risk = self.chart_generator.injury_risk_chart()
        chart_goals = self.chart_generator.goals_by_season()

        high_risk = int(injury['injury_risk'].sum())
        low_risk = int((injury['injury_risk'] == 0).sum())

        msg = "## 📊 Platform Overview — AthleteIQ Analytics\n\n"
        msg += f"### Database Summary\n"
        msg += f"- 👥 **Players:** {len(perf):,}\n"
        msg += f"- ⚽ **Matches:** {len(matches):,}\n"
        msg += f"- 🏟️ **Teams:** {len(teams)}\n"
        msg += f"- 🥅 **Total Goals:** {int(matches['total_goals'].sum()):,}\n"
        msg += f"- ⚽ **Avg Goals/Match:** {matches['total_goals'].mean():.2f}\n\n"
        msg += f"### Player Performance\n"
        msg += f"- Avg Rating: **{perf['overall_rating'].mean():.1f}** | Avg Performance Score: **{perf['performance_score'].mean():.1f}**\n"
        msg += f"- Top Performer: **{perf.loc[perf['performance_score'].idxmax(), 'player_name']}** ({perf['performance_score'].max():.1f})\n\n"
        msg += f"### Injury Risk Distribution\n"
        msg += f"- 🔴 High Risk: **{high_risk}** ({high_risk / len(injury) * 100:.1f}%)\n"
        msg += f"- 🟢 Low Risk: **{low_risk}** ({low_risk / len(injury) * 100:.1f}%)\n"

        return {
            'type': 'overview',
            'message': msg,
            'charts': [chart_risk, chart_goals],
            'stats': {
                'total_players': len(perf), 'total_matches': len(matches),
                'total_teams': len(teams), 'total_goals': int(matches['total_goals'].sum()),
                'high_risk_players': high_risk, 'low_risk_players': low_risk,
            },
            'suggestions': ["Top 10 players", "Show anomalies", "Squad fatigue overview", "Show declining players"]
        }

    def _handle_head_to_head(self, entities, message):
        if len(entities) < 2:
            return self._error_response("Please specify two teams")
        return self._handle_match_prediction(entities, message)

    def _handle_coaching(self, entities, message):
        name = self._extract_player_name(entities, message)
        player_id = self._find_player_id(name)
        if not player_id:
            return self._player_not_found(name)

        analysis = self.perf_analyzer.analyze_player(player_id)
        injury = self.injury_predictor.predict_with_explanation(player_id)
        fatigue = self.fatigue_monitor.get_player_fatigue(player_id)

        if not analysis:
            return self._player_not_found(name)

        p = analysis['player']
        msg = f"## 🧑‍🏫 Personalized Coaching Plan: **{p['name']}**\n\n"

        # Strengths to maintain
        msg += "### ✅ Maintain Strengths\n"
        for s in analysis['strengths'][:3]:
            msg += f"- **{s['skill']}** ({s['value']:.0f}/100) — Continue current training intensity\n"

        # Weaknesses to improve
        msg += "\n### 📈 Priority Development Areas\n"
        for i, w in enumerate(analysis['weaknesses'][:3]):
            priority = ['🔴 HIGH', '🟠 MEDIUM', '🟡 MODERATE'][min(i, 2)]
            msg += f"- {priority}: **{w['skill']}** ({w['value']:.0f}/100)\n"

            # Specific drills
            drills = {
                'Finishing': '  - Shooting drills: 30 shots/session (near/far post)\n  - 1v1 scenarios with keeper\n',
                'Heading': '  - Aerial crossing drills\n  - Jumping power exercises\n',
                'Standing Tackle': '  - Defensive 1v1 scenarios\n  - Timing and positioning drills\n',
                'Marking': '  - Shadow marking exercises\n  - Positional awareness sessions\n',
                'Crossing': '  - Wing play scenarios\n  - Delivery accuracy under pressure\n',
                'Sprint Speed': '  - 30m sprint intervals\n  - Resistance sprints\n',
                'Stamina': '  - High-intensity interval training\n  - Endurance runs (5-10km)\n',
                'Strength': '  - Weight training (squats, deadlifts)\n  - Core stability exercises\n',
            }
            msg += drills.get(w['skill'], '  - Targeted skill-specific drills\n  - Video analysis sessions\n')

        # Physical conditioning
        msg += "\n### 💪 Physical Conditioning\n"
        phys = analysis['scores']['physical']
        if phys < 60:
            msg += "- 🔴 **Physical conditioning needs significant improvement**\n"
            msg += "- 3x/week strength training + 2x/week cardio sessions\n"
        elif phys < 75:
            msg += "- 🟡 **Moderate physical conditioning** — room for growth\n"
            msg += "- 2x/week strength + 2x/week sport-specific conditioning\n"
        else:
            msg += "- 🟢 **Strong physical base** — maintain current program\n"

        # Injury management
        if injury and injury['risk_level'] == 'High':
            msg += "\n### 🏥 Injury Prevention Protocol\n"
            msg += f"- ⚠️ **High injury risk detected** — adjusted training mandatory\n"
            for f in injury.get('risk_factors', [])[:3]:
                msg += f"- {f}\n"
            msg += "- Reduce training load by 30%\n"
            msg += "- Daily physiotherapy and recovery sessions\n"
            msg += "- Pre-session activation routine (15 min)\n"

        # Fatigue management
        if fatigue and fatigue['fatigue_level'] in ('High', 'Critical'):
            msg += f"\n### 🔋 Fatigue Management\n"
            msg += f"- Current fatigue: **{fatigue['fatigue_level']}** ({fatigue['fatigue_index']:.1f}/10)\n"
            msg += f"- {fatigue['recommendation']}\n"

        msg += f"\n### 📅 Weekly Schedule Recommendation\n"
        msg += "| Day | Morning | Afternoon |\n"
        msg += "|-----|---------|----------|\n"
        msg += "| Mon | Technical drills | Strength training |\n"
        msg += "| Tue | Tactical session | Skill-specific work |\n"
        msg += "| Wed | Match preparation | Active recovery |\n"
        msg += "| Thu | Set piece practice | Physical conditioning |\n"
        msg += "| Fri | Light training | Pre-match prep |\n"
        msg += "| Sat | **Match Day** | — |\n"
        msg += "| Sun | Rest & Recovery | Physiotherapy |\n"

        return {
            'type': 'coaching_plan',
            'message': msg,
            'analysis': analysis,
            'suggestions': [
                f"Analyze {p['name']}",
                f"Injury risk for {p['name']}",
                "Show improving players",
            ]
        }

    def _handle_general(self, entities, message):
        # Try data engine as fallback
        result = self.data_engine.query(message)
        if result and result.get('type') != 'no_results':
            msg = self._format_data_result(result)
            return {
                'type': result['type'],
                'message': msg,
                'data': result.get('data'),
                'suggestions': ["Show stats overview", "Top 10 players", "Detect anomalies", "Squad fatigue"]
            }

        # Truly unknown query
        return {
            'type': 'help',
            'message': "## 🤖 AthleteIQ Sports Analyst\n\nI can help you with:\n\n"
                       "- 🔍 **Player Analysis**: 'Tell me about Messi'\n"
                       "- ⚔️ **Comparisons**: 'Compare Messi vs Ronaldo'\n"
                       "- 🏥 **Injury Risk**: 'Injury risk for Neymar'\n"
                       "- 🔋 **Fatigue**: 'Fatigue report for Hazard'\n"
                       "- 📈 **Rankings**: 'Top 10 players by attack'\n"
                       "- 📉 **Trends**: 'Show declining players'\n"
                       "- ⚽ **Predictions**: 'Predict Barcelona vs Real Madrid'\n"
                       "- 🧑‍🏫 **Coaching**: 'Coaching plan for Ronaldo'\n"
                       "- 🏟️ **Lineup**: 'Recommend lineup for Barcelona'\n"
                       "- 🚨 **Anomalies**: 'Detect anomalies'\n"
                       "- 📊 **Stats**: 'Show overall statistics'\n",
            'suggestions': ["Top 10 players", "Show stats overview", "Detect anomalies", "Show declining players"]
        }

    # ─────────── HELPERS ───────────

    def _extract_player_name(self, entities, message):
        for e in entities:
            if e and not e.isdigit():
                return e.strip().rstrip('?.,! ')
        # Try to find a name in message
        words = message.split()
        name_parts = [w for w in words if len(w) > 2 and w[0].isupper() and w.lower() not in
                     {'the', 'who', 'what', 'how', 'why', 'show', 'tell', 'about', 'for', 'with', 'and', 'from'}]
        return ' '.join(name_parts) if name_parts else message.strip()

    def _extract_team_name(self, message):
        words = message.split()
        name_parts = [w for w in words if w[0].isupper() and w.lower() not in
                     {'team', 'show', 'tell', 'about', 'the', 'analysis', 'info'}]
        return ' '.join(name_parts) if name_parts else None

    def _find_player_id(self, name):
        if not name:
            return None
        perf = self.etl.gold['fact_player_performance']
        matches = perf[perf['player_name'].str.lower().str.contains(name.lower(), na=False)]
        if matches.empty:
            return None
        return int(matches.iloc[0]['player_api_id'])

    def _player_not_found(self, name):
        perf = self.etl.gold['fact_player_performance']
        similar = perf[perf['player_name'].str.lower().str.contains(name[:3].lower() if len(name) >= 3 else name.lower(), na=False)].head(5)
        suggestions = similar['player_name'].tolist() if not similar.empty else []

        msg = f"⚠️ Could not find player **'{name}'** in the database.\n\n"
        if suggestions:
            msg += "Did you mean:\n"
            for s in suggestions:
                msg += f"- {s}\n"

        return {
            'type': 'error',
            'message': msg,
            'suggestions': [f"Tell me about {s}" for s in suggestions[:3]] if suggestions else ["Top 10 players", "Show stats"]
        }

    def _error_response(self, error_msg):
        return {
            'type': 'error',
            'message': f"⚠️ {error_msg}",
            'suggestions': ["Show stats overview", "Top 10 players", "Help"]
        }

    def _format_data_result(self, result):
        if result['type'] == 'player_ranking':
            msg = f"## 🏆 Top Players by {result['metric'].replace('_', ' ').title()}\n\n"
            for i, p in enumerate(result['data'][:10]):
                msg += f"{i+1}. **{p['player_name']}** — {p.get(result['metric'], 'N/A')}\n"
            if result.get('sql'):
                msg += f"\n```sql\n{result['sql']}\n```\n"
            return msg
        elif result['type'] == 'counts':
            msg = "## 📊 Database Statistics\n\n"
            for k, v in result['data'].items():
                msg += f"- **{k.replace('_', ' ').title()}:** {v:,}\n"
            return msg
        elif result['type'] == 'statistics':
            d = result['data']
            msg = f"## 📊 {d['attribute'].replace('_', ' ').title()} Statistics\n\n"
            msg += f"- Mean: **{d['mean']}** | Median: **{d['median']}**\n"
            msg += f"- Std Dev: {d['std']} | Min: {d['min']} | Max: {d['max']}\n"
            return msg
        else:
            return str(result.get('data', ''))

    def _df_to_list(self, df):
        records = df[['player_name', 'overall_rating', 'performance_score', 'age']].to_dict('records')
        cleaned = []
        for r in records:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = round(float(v), 1)
                else:
                    clean[k] = v
            cleaned.append(clean)
        return cleaned
