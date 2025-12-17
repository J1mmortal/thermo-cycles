import math
from numpy import log as ln
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
            U = self.Cv * T
            H = self.Cp * T
            S = self.Cp * ln(T/273.15) - self.R * ln(P/(10**5))
            st.set(V=V, U=U, H=H, S=S)
        if st.has("T", "V"):
            T = st.get("T")
            V = st.get("V")
            P = self.R * T / V
            st.set(P=P)
            U = self.Cv * T
            H = self.Cp * T
            S = self.Cv * ln(T/273.15) + self.R * ln(V/(self.R * 273.15 / (10**5)))
            st.set(P=P, U=U, H=H, S=S)
        if st.has("P", "V"):
            P = st.get("P")
            V = st.get("V")
            T = P * V / self.R
            U = self.Cv * T
            H = self.Cp * T
            S = self.Cp * ln(T/273.15) - self.R * ln(P/(10**5))     
            st.set(T=T, U=U, H=H, S=S)
            
    def isentropic_from(self, s1: State, s2: State, known_prop: str) -> None:
        known_prop = known_prop.upper()
        T1 = s1.get("T")
        P1 = s1.get("P")
        V1 = s1.get("V")
        S1 = s1.get("S")
        S2 = S1
        
        if known_prop == "V":
            V2 = s2.get("V")
            P2 = P1 * (V1 / V2) ** self.gamma
            T2 = T1 * (V1 / V2) ** (self.gamma - 1)
            U2 = self.Cv * T2
            H2 = self.Cp * T2
            s2.set(P=P2, T=T2, U=U2, H=H2, S=S2)
        
        elif known_prop == "T":
            T2 = s2.get("T")
            V2 = V1 * (T1 / T2) ** (1/(self.gamma - 1))
            P2 = P1 * (V1 / V2) ** self.gamma
            U2 = self.Cv * T2
            H2 = self.Cp * T2
            s2.set(P=P2, V=V2, U=U2, H=H2, S=S2)
            
        elif known_prop == "P":
            P2 = s2.get("P")
            V2 = V1 * (P1 / P2) ** (1/self.gamma)
            T2 = T1 * (V1 / V2) ** (self.gamma - 1)
            U2 = self.Cv * T2
            H2 = self.Cp * T2
            s2.set(V=V2, T=T2, U=U2, H=H2, S=S2)

        else:
            raise ValueError(f"Unknown known_prop={known_prop!r}")

    def isochoric_from(self, s1: State, s2: State, known_prop: str) -> None:
        V1 = s1.get("V")
        V2 = V1
        
        if known_prop == "T":
            T2 = s2.get("T")
            P2 = self.R * T2 / V2
            U2 = self.Cv * T2
            H2 = self.Cp * T2
            S2 = self.Cv * ln(T2/273.15) + self.R * ln(V2/(self.R * 273.15 / (10**5)))
            s2.set(P=P2, V=V2, U=U2, H=H2, S=S2)
        
        elif known_prop == "P":
            P2 = s2.get("P")
            T2 = P2 * V2 / self.R
            U2 = self.Cv * T2
            H2 = self.Cp * T2
            S2 = self.Cv * ln(T2/273.15) + self.R * ln(V2/(self.R * 273.15 / (10**5)))
            s2.set(T=T2, V=V2, U=U2, H=H2, S=S2)
        
        else:
            raise ValueError(f"Unknown known_prop={known_prop!r}")

    def isobaric_from(self, s1: State, s2: State, known_prop: str) -> None:
        P1 = s1.get("P")
        P2 = P1