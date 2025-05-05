import yaml

import yaml
import datetime

class GlobalRules:
    def __init__(self, rules=None):
        # rules: dict or list of rules
        self.rules = rules.get('rules', []) if isinstance(rules, dict) else (rules or [])
        # In-memory state for rate limiting and denial history
        self._call_history = {}
        self._denial_history = {}

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            rules = yaml.safe_load(f)
        return cls(rules)

    def check(self, action, context=None):
        import time
        context = context or {}
        now = time.time()
        for rule in self.rules:
            if rule.get('action') != action:
                continue
            for cond in rule.get('conditions', []):
                ctype = cond.get('type')
                if ctype == 'confidence_threshold':
                    min_conf = cond.get('min_confidence', 0.0)
                    modalities = cond.get('modalities', [])
                    for m in modalities:
                        conf = context.get(m, None)
                        if conf is not None and conf < min_conf:
                            self._record_denial(action, now)
                            return False, rule.get('message', 'Confidence too low')
                elif ctype == 'fuzzy_confidence':
                    weighted = cond.get('weighted', False)
                    weights = cond.get('weights', {})
                    min_avg = cond.get('min_average', 0.0)
                    confs = []
                    if weighted and weights:
                        total = 0.0
                        wsum = 0.0
                        for m, w in weights.items():
                            c = context.get(m, None)
                            if c is not None:
                                total += float(c) * float(w)
                                wsum += float(w)
                        avg = total / wsum if wsum > 0 else 0.0
                    else:
                        # Unweighted average
                        vals = [float(context.get(m, 0.0)) for m in weights.keys() or context.keys()]
                        avg = sum(vals) / len(vals) if vals else 0.0
                    if avg < min_avg:
                        self._record_denial(action, now)
                        return False, rule.get('message', 'Average confidence too low')
                elif ctype == 'rate_limit':
                    max_calls = cond.get('max_calls', 5)
                    per_seconds = cond.get('per_seconds', 60)
                    history = self._call_history.setdefault(action, [])
                    # Remove old calls
                    history = [t for t in history if now - t < per_seconds]
                    if len(history) >= max_calls:
                        self._call_history[action] = history
                        self._record_denial(action, now)
                        return False, rule.get('message', 'Too many requests')
                    history.append(now)
                    self._call_history[action] = history
                elif ctype == 'recent_denials':
                    max_denials = cond.get('max_denials', 2)
                    window = cond.get('window_seconds', 300)
                    denials = self._denial_history.get(action, [])
                    denials = [t for t in denials if now - t < window]
                    if len(denials) >= max_denials:
                        self._denial_history[action] = denials
                        return False, rule.get('message', 'Too many recent denials')
                    self._denial_history[action] = denials
                elif ctype == 'user_role':
                    allowed = cond.get('allowed_roles', [])
                    user_role = context.get('user_role', None)
                    if user_role not in allowed:
                        self._record_denial(action, now)
                        return False, rule.get('message', 'User role not allowed')
                elif ctype == 'time_restriction':
                    allowed_hours = cond.get('allowed_hours', [])
                    now_hour = self._get_current_hour(context)
                    if now_hour is not None:
                        if isinstance(allowed_hours, list) and len(allowed_hours) == 2:
                            start, end = allowed_hours
                            if not (start <= now_hour < end):
                                self._record_denial(action, now)
                                return False, rule.get('message', 'Action not allowed at this time')
        return True, "No global rule violations."

    def _record_denial(self, action, now):
        history = self._denial_history.setdefault(action, [])
        history.append(now)
        self._denial_history[action] = history

    def _get_current_hour(self, context):
        # Use context['current_time'] if provided, else system time
        if 'current_time' in context:
            try:
                dt = datetime.datetime.fromisoformat(context['current_time'])
                return dt.hour
            except Exception:
                pass
        try:
            return datetime.datetime.now().hour
        except Exception:
            return None
