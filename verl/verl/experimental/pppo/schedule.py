"""Serializable, idempotent validation-driven prefix schedule."""
from dataclasses import dataclass, asdict
import math


@dataclass
class PrefixSchedule:
    eta: int = 15
    best: float | None = None
    bad_count: int = 0
    cooldown: int = 0
    last_validation_step: int = -1
    next_effective_step: int = 0

    def observe(self, step, accuracy, *, kind="plateau", patience=3, min_delta=0., cooldown_validations=1):
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("invalid holdout accuracy")
        if step <= self.last_validation_step:
            return False
        self.last_validation_step = step
        if self.best is None or step == 0:
            self.best = accuracy
            return False
        if kind == "fixed":
            return False
        if self.cooldown:
            self.cooldown -= 1
            self.best = max(self.best, accuracy)
            return False
        if accuracy > self.best + min_delta:
            self.best, self.bad_count = accuracy, 0
        else:
            self.bad_count += 1
        if self.bad_count >= patience and self.eta < 35:
            self.eta += 5
            self.best, self.bad_count = accuracy, 0
            self.cooldown = cooldown_validations
            self.next_effective_step = step + 1
            return True
        return False

    def state_dict(self):
        return asdict(self)
