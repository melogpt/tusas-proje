from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

@dataclass
class ParamSpec:
    key: str
    label: str
    unit: str
    vmin: float
    vmax: float
    caution_hi: Optional[float] = None
    warning_hi: Optional[float] = None
    caution_lo: Optional[float] = None
    warning_lo: Optional[float] = None
    orientation: str = "V"
    decimals: int = 1
    min_warning_s: int = 5
    min_caution_s: int = 4

@dataclass
class WcaEntry:
    key: str
    text: str
    severity: str
    first_seen_ms: int
    last_seen_ms: int
    count: int = 1

class WcaStore:
    def __init__(self):
        self.entries: List[WcaEntry] = []
        self._by: Dict[str, WcaEntry] = {}

    @staticmethod
    def rank(sev: str) -> int:
        return {"WARNING": 3, "CAUTION": 2, "ADVISORY": 1}.get(sev, 0)

    def upsert(self, now_ms: int, sev: str, key: str, text: str):
        if key in self._by:
            e = self._by[key]
            if self.rank(sev) > self.rank(e.severity):
                e.severity = sev
            e.text = text
            e.last_seen_ms = now_ms
            e.count += 1
        else:
            e = WcaEntry(key, text, sev, now_ms, now_ms, 1)
            self.entries.append(e)
            self._by[key] = e

    def snapshot_sorted(self) -> List[WcaEntry]:
        return sorted(self.entries, key=lambda e: (self.rank(e.severity), e.last_seen_ms, e.first_seen_ms), reverse=True)

    def counts(self) -> Tuple[int, int, int]:
        w = sum(1 for e in self.entries if e.severity == "WARNING")
        c = sum(1 for e in self.entries if e.severity == "CAUTION")
        a = sum(1 for e in self.entries if e.severity == "ADVISORY")
        return w, c, a

@dataclass
class FaultState:
    active: bool = False
    start_ms: int = 0
    last_ms: int = 0

class FaultGate:
    def __init__(self):
        self._s: Dict[str, FaultState] = {}

    def update(self, key: str, now_ms: int, active: bool) -> FaultState:
        st = self._s.get(key)
        if st is None:
            st = FaultState()
            self._s[key] = st
        if not active:
            st.active = False
            st.start_ms = 0
            st.last_ms = now_ms
            return st
        if not st.active:
            st.active = True
            st.start_ms = now_ms
            st.last_ms = now_ms
            return st
        st.last_ms = now_ms
        return st