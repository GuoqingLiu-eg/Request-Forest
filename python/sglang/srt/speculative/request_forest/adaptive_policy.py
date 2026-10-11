from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    enter_rf: int = 64
    exit_rf: int = 48
    confirm_rounds: int = 2
    min_dwell_rounds: int = 4

    def __post_init__(self):
        for name in ('enter_rf', 'exit_rf', 'confirm_rounds', 'min_dwell_rounds'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.exit_rf >= self.enter_rf:
            raise ValueError('exit_rf must be smaller than enter_rf')


@dataclass(frozen=True)
class Decision:
    mode: str
    active_batch: int
    phase: str
    reason: str


class AdaptiveRFPolicy:


    def __init__(self, config=None):
        self.config = config if config is not None else PolicyConfig()
        if not isinstance(self.config, PolicyConfig):
            raise TypeError('config must be PolicyConfig')
        self._mode = 'dflash2'
        self._streak = 0
        self._dwell = 0
        self._pending = None

    @property
    def mode(self):
        return self._mode

    def prepare_load(self, active_batch, *, exact=True):

        if self._pending is not None:
            raise RuntimeError('Cannot change load mode while a round is outstanding')
        if type(active_batch) is not int or active_batch < 0:
            raise ValueError('active_batch must be a nonnegative integer')
        if type(exact) is not bool:
            raise ValueError('exact must be a boolean')
        if active_batch >= self.config.enter_rf and self._mode != 'rf':
            self._mode = 'rf'
            self._streak = 0
            self._dwell = self.config.min_dwell_rounds
            return True
        if exact and active_batch <= 1 and self._mode != 'dflash2':
            self._mode = 'dflash2'
            self._streak = 0
            self._dwell = self.config.min_dwell_rounds
            return True
        return False

    def propose(self, active_batch, phase='decode'):
        if self._pending is not None:
            raise RuntimeError('A round is already outstanding')
        if type(active_batch) is not int or active_batch < 0:
            raise ValueError('active_batch must be a nonnegative integer')
        if phase not in ('decode', 'prefill', 'idle'):
            raise ValueError('phase must be decode, prefill or idle')
        switched = phase == 'decode' and self.prepare_load(active_batch)
        qualifies = self._mode == 'rf' and active_batch <= self.config.exit_rf
        reason = ('no_decode_observation' if phase != 'decode' or active_batch == 0
                  else 'immediate_singleton' if switched and self._mode == 'dflash2'
                  else 'immediate_high_load' if switched
                  else 'confirming_switch' if qualifies else 'retain_mode')
        decision = Decision(self._mode, active_batch, phase, reason)
        self._pending = decision
        return decision

    def _require_pending(self, decision):
        if self._pending is None or decision is not self._pending:
            raise RuntimeError('Decision does not own the outstanding round')

    def commit(self, decision):
        self._require_pending(decision)
        self._pending = None
        if decision.phase != 'decode' or decision.active_batch == 0:
            return False
        qualifies = self._mode == 'rf' and decision.active_batch <= self.config.exit_rf
        self._streak = min(self.config.confirm_rounds, self._streak + 1) if qualifies else 0
        self._dwell = max(0, self._dwell - 1)
        if qualifies and self._streak >= self.config.confirm_rounds and self._dwell == 0:
            self._mode = 'dflash2'
            self._streak = 0
            self._dwell = self.config.min_dwell_rounds
            return True
        return False

    def abort(self, decision):
        self._require_pending(decision)
        self._pending = None
