from core.state import State
from core.cycle_state import CycleState
from core.ideal_gas import IdealGasCycleBase

class AtkinsonSolver(IdealGasCycleBase):
    def __init__(self, gamma: float = 1.4, R: float = 287.0):
        super().__init__(gamma=gamma, R=R)

    def solve(self, cs: CycleState) -> None:
        if cs.cycle_type.lower() != "atkinson":
            raise ValueError(f"AtkinsonSolver got cycle_type={cs.cycle_type}")

        mode = cs.mode
        internal = cs.internal

        if mode == "design":
            T1 = internal["T1"]
            P1 = internal["P1"]
            r_c = internal["R_C"]
            T3 = internal.get("T3")
            q_in = internal.get("Q_IN")
            
            if q_in is not None:
                s1 = cs.add_state(1, T=T1, P=P1)
                self.compute_state_properties(s1)
                V1 = s1.get("V")
                V2 = V1 / r_c
                s2 = cs.add_state(2, V=V2)
                self.isentropic_from(s1, s2, "V")
                P2 = s2.get("P")
                s3 = cs.add_state(3, P=P2)
                self.heat_supply(s2, s3, q_in)
                self.isobaric_from(s2, s3, "T")
                V4 = V1
                s4 = cs.add_state(4, V=V4)
                self.isentropic_from(s3, s4, "V")
            elif T3 is not None:
                s1 = cs.add_state(1, T=T1, P=P1)
                self.compute_state_properties(s1)
                V1 = s1.get("V")
                V2 = V1 / r_c
                s2 = cs.add_state(2, V=V2)
                self.isentropic_from(s1, s2, "V")
                P2 = s2.get("P")
                s3 = cs.add_state(3, P=P2, T=T3)
                self.isobaric_from(s2, s3, "T")
                V4 = V1
                s4 = cs.add_state(4, V=V4)
                self.isentropic_from(s3, s4, "V")
            T4 = s4.get("T")
            T3 = s3.get("T")
            T2 = s2.get("T")
            q_23 = self.Cp * (T3 - T2)
            q_41 = self.Cv * (T4 - T1)
            eta_th = 1 - q_41 / q_23
            w_net = eta_th * q_23
            MEP = w_net / (V1 - V2) if V1 > V2 else 0.0
            cs.results["eta_th"] = eta_th
            cs.results["w_net"] = w_net
            cs.results["MEP"] = MEP
            
        elif mode == "efficiency":
            T1 = internal["T1"]
            T3_MAX = internal["T3_MAX"]
            r_c_min = internal["R_C_MIN"]
            r_c_max = internal["R_C_MAX"]
            P_ratio = internal["P_RATIO"]
            i = 0
            eta_max = 0
            steps = 10000
            for k in range(steps + 1):
                r_c = r_c_min + k * (r_c_max - r_c_min) / steps
                P1 = 101325
                s1 = cs.add_state(1, T=T1, P=P1)
                self.compute_state_properties(s1)
                V1 = s1.get("V")
                V2 = V1 / r_c
                s2 = cs.add_state(2, V=V2)
                self.isentropic_from(s1, s2, "V")
                P2 = s2.get("P")
                if P2/P1 > P_ratio:
                    continue
                s3 = cs.add_state(3, P=P2, T=T3_MAX)
                self.isobaric_from(s2, s3, "T")
                V4 = V1
                s4 = cs.add_state(4, V=V4)
                self.isentropic_from(s3, s4, "V")
                T2 = s2.get("T")
                T3 = s3.get("T")
                T4 = s4.get("T")
                q_23 = self.Cp * (T3 - T2)
                q_41 = self.Cv * (T4 - T1)
                eta = 1 - q_41 / q_23
                if eta > eta_max:
                    i_max = i
                    eta_max = eta
                    r_c_opt = r_c
                i += 1
            cs.results["eta_th"] = eta_max
            cs.results["optimal_r_c"] = r_c_opt
            
                
                
                
        else:
            raise NotImplementedError(f"Atkinson mode {mode} not implemented yet")

def test_atkinson_design():
    '''design mode test
    cs = CycleState(
        cycle_type="atkinson",
        mode="design",
        known={},           # can be empty for this manual test
        internal={
            "T1": 298.0,         #K
            "P1": 100000.0,      #Pa
            "R_C": 20.0,         #compression ratio
            "T3": 2480.988       #K
            #"Q_IN": 1500000,    #J/kg, example heat supply
        },
    )

    # 2. Solve
    solver = AtkinsonSolver(gamma=1.4, R=287.0)
    solver.solve(cs)

    # 3. Print all states
    print("=== Atkinson design test ===")
    for sid, st in sorted(cs.states.items()):
        print(f"\nState {sid}:")
        for k, v in st.properties.items():
            print(f"  {k} = {v:.3f}")

    # 4. Optionally print any results you compute later
    print("\nResults dict:", cs.results)
    '''
    cs = CycleState(
        cycle_type="atkinson",
        mode="efficiency",
        known={},           # can be empty for this manual test
        internal={
            "T1": 298.0,             #K
            "T3_MAX": 2700.0,        #Pa
            "R_C_MIN": 5.0,          #compression ratio
            "R_C_MAX": 60.0,         #compression ratio
            "P_RATIO": 70            #max pressure ratio between P2/P1
        },
    )

    # 2. Solve
    solver = AtkinsonSolver(gamma=1.4, R=287.0)
    solver.solve(cs)

    # 3. Print all states
    print("=== Atkinson design test ===")
    for sid, st in sorted(cs.states.items()):
        print(f"\nState {sid}:")
        for k, v in st.properties.items():
            print(f"  {k} = {v:.3f}")

    # 4. Optionally print any results you compute later
    print("\nResults dict:", cs.results)
if __name__ == "__main__":
    test_atkinson_design()
    