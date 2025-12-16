from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class State:
    fluid: str = "air"
    gamma: float = 1.4
    R: float = 287.0
    state_id: Optional[int] = None
    properties: Dict[str, float] = field(default_factory=dict)

    def set(self, **kwargs: float) -> "State":
        for k, v in kwargs.items():
            self.properties[k.upper()] = float(v)
        return self

    def get(self, name: str) -> float:
        name = name.upper()
        if name not in self.properties:
            raise KeyError(f"Property {name} not set in state {self.state_id}")
        return self.properties[name]

    def has(self, *names: str) -> bool:
        return all(n.upper() in self.properties for n in names)

    def count(self) -> int:
        return len(self.properties)

    def summary(self) -> str:
        props = ", ".join(f"{k}={v:.3g}" for k, v in self.properties.items())
        sid = f"#{self.state_id}" if self.state_id is not None else ""
        return f"State{sid}({props})"

    def __repr__(self) -> str:
        return self.summary()
