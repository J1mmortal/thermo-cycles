import math
from core.state import State

class IdealGasCycleBase:
    def __init__(self, gamma: float = 1.4, R: float = 287.0):
        self.gamma = gamma
        self.R = R
        self.Cp = gamma * R / (gamma - 1.0)
        self.Cv = R / (gamma - 1.0)

    def compute_state_properties(self, st: State) -> None:
        if st.has("T", "P"):
            T = st.get("T")
            P = st.get("P")
            V = self.R * T / P
            st.set(V=V)
            # You can also set h, u, s here if you want

    def isentropic_from(self, s1: State, s2: State, known_prop: str) -> None:
        known_prop = known_prop.upper()
        #TODO: implement real relations
        raise NotImplementedError

    def isochoric_from(self, s1: State, s2: State, known_prop: str) -> None:
        #TODO: implement real relations
        raise NotImplementedError

    def isobaric_from(self, s1: State, s2: State, known_prop: str) -> None:
        #TODO: implement real relations
        raise NotImplementedError
