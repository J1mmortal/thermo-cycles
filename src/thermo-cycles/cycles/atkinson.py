from core.state import State
from core.cycle_state import CycleState
from core.ideal_gas import IdealGasCycleBase

class AtkinsonSolver(IdealGasCycleBase):
    def __init__(self, gamma: float = 1.4, R: float = 287.0):
        super().__init__(gamma=gamma, R=R)

    def solve(self, cs: CycleState) -> None:
        """
        Fill cs.states and cs.results based on:
        - cs.cycle_type == "atkinson"
        - cs.mode
        - cs.internal (canonical inputs)
        """
        if cs.cycle_type.lower() != "atkinson":
            raise ValueError(f"AtkinsonSolver got cycle_type={cs.cycle_type}")

        mode = cs.mode
        internal = cs.internal

        # Example: handle design mode with T1,P1,r_c,r_e and T3 or Q_in
        if mode == "design":
            T1 = internal["T1"]
            P1 = internal["P1"]
            r_c = internal["R_C"]
            r_e = internal["R_E"]
            T3 = internal.get("T3")
            Q_in = internal.get("Q_IN")

            # --- YOUR MATH GOES HERE ---
            # 1) create and fill state 1
            # 2) use isentropic_from etc. to get state 2
            # 3) isochoric 2->3 with T3 or Q_in
            # 4) isentropic 3->4 with r_e
            # 5) compute Q_in, Q_out, W_net, eta, etc.

            # Example placeholders (remove when you implement):
            s1 = cs.add_state(1, T=T1, P=P1)
            self.compute_state_properties(s1)
            # ...
            cs.results["efficiency"] = 0.4  # placeholder

        elif mode == "efficiency":
            # Here you'd likely:
            # - loop/optimize over r_c,r_e
            # - call some internal core solver repeatedly
            # - choose the best combination and save states/results
            raise NotImplementedError("Atkinson efficiency mode not implemented yet")

        else:
            raise NotImplementedError(f"Atkinson mode {mode} not implemented yet")
