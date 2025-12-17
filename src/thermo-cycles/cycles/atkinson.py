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
            q_23 = q_in
            q_41 = self.Cv * (T4 - T1)
            eta_th = 1 - q_41 / q_23
            w_net = eta_th * q_in
            cs.results["eta_th"] = eta_th
            cs.results["w_net"] = w_net
            
        elif mode == "efficiency":

            raise NotImplementedError("Atkinson efficiency mode not implemented yet")

        else:
            raise NotImplementedError(f"Atkinson mode {mode} not implemented yet")

def test_atkinson_design():
    # 1. Build a CycleState with "design" mode and internal values
    cs = CycleState(
        cycle_type="atkinson",
        mode="design",
        known={},           # can be empty for this manual test
        internal={
            "T1": 298.0,    # K
            "P1": 100000.0, # Pa
            "R_C": 20.0,    # compression ratio
            "Q_IN": 1500000,    # J/kg, example heat supply
            # or use "T3" instead of q_IN if your solver supports that
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
    