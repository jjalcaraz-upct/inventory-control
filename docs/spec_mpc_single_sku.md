# MILP Formulation for a Single-SKU Inventory Control MPC Problem

This document specifies **precisely and unambiguously** the mathematical model that must be translated into **Pyomo code**. The goal is to construct a **MILP** corresponding to a **finite-horizon MPC inventory optimization with scenario sampling**.

The implementation should be modular so that the **horizon length, number of scenarios, and scenario trajectories can be easily changed**.

---

# 1. Sets

Define the following sets.

- Time steps (MPC horizon):

$$
K = \{0,1,\dots,H-1\}
$$

- Inventory states are defined at:

$$
K^+ = \{0,1,\dots,H\}
$$

- Lead time pipeline positions:

$$
L = \{1,\dots,L_{\max}\}
$$

- Scenario index:

$$
S = \{1,\dots,N_s\}
$$

---

# 2. Parameters

All parameters are assumed **given constants**.

### Horizon parameters

- $H$ : MPC horizon length
- $L_{\max}$ : maximum lead time

### Cost parameters

- $K_{fix}$ : fixed ordering cost
- $v$ : variable ordering cost per unit
- $h$ : holding cost per unit per period
- $p$ : lost-sales penalty per unit
- $\lambda$ : terminal penalty weight

### Ordering limits

- $q^{min}$
- $q^{max}$

### Initial state

- $I_0$ : initial on-hand inventory
- $P_0(\ell)$ : initial pipeline inventory arriving in $\ell$ periods

### Scenario data

For each scenario $s \in S$ and period $k \in K$:

- $D_{k,s}$ : demand realization
- $L_{k,s}$ : lead time realization for order placed at time $k$

Define the indicator constant

$$
\delta_{k,s,\ell} =
\begin{cases}
1 & \text{if } L_{k,s} = \ell \\
0 & \text{otherwise}
\end{cases}
$$

### Target inventory

- $I^{target}$

---

# 3. Decision Variables

## Order decisions

For each $k \in K$:

- $q_k \ge 0$ : order quantity
- $z_k \in \{0,1\}$ : binary variable indicating whether an order is placed

Ordering constraints:

$$
q^{min} z_k \le q_k \le q^{max} z_k
$$

---

## State variables (scenario dependent)

For each $k \in K^+$, $s \in S$:

- $I_{k,s} \ge 0$ : inventory at start of period

For each $k \in K^+$, $s \in S$, $\ell \in L$:

- $P_{k,s,\ell} \ge 0$ : pipeline inventory arriving in $\ell$ periods

---

## Auxiliary variables

For each $k \in K$, $s \in S$:

- $R_{k,s} \ge 0$ : arrivals
- $\ell_{k,s} \ge 0$ : lost sales

Terminal absolute deviation:

For each $s \in S$:

- $u_s \ge 0$

---

# 4. Initial Conditions

For all scenarios $s$:

$$
I_{0,s} = I_0
$$

$$
P_{0,s,\ell} = P_0(\ell) \quad \forall \ell \in L
$$

---

# 5. System Dynamics

For all $k \in K$, $s \in S$:

## Arrivals

$$
R_{k,s} = P_{k,s,1}
$$

---

## Lost sales linearization

$$
\ell_{k,s} \ge D_{k,s} - (I_{k,s} + R_{k,s})
$$

$$
\ell_{k,s} \ge 0
$$

---

## Inventory transition

$$
I_{k+1,s} =
I_{k,s} + R_{k,s} - D_{k,s} + \ell_{k,s}
$$

---

# 6. Pipeline Dynamics

Define intermediate shifted pipeline:

$$
\tilde P_{k+1,s,\ell} =
\begin{cases}
P_{k,s,\ell+1} & \ell < L_{max} \\
0 & \ell = L_{max}
\end{cases}
$$

Injection of new order:

$$
P_{k+1,s,\ell} =
\tilde P_{k+1,s,\ell} + q_k \cdot \delta_{k,s,\ell}
$$

---

# 7. Terminal Cost Linearization

Absolute deviation from target inventory:

$$
u_s \ge I_{H,s} - I^{target}
$$

$$
u_s \ge I^{target} - I_{H,s}
$$

---

# 8. Objective Function

Minimize expected cost across scenarios:

$$
\min
\sum_{k \in K} \left(K_{fix} z_k + v q_k \right)
+
\frac{1}{N_s}
\sum_{s \in S}
\left[
\sum_{k \in K}
\left(
h I_{k,s} + p \ell_{k,s}
\right)
+
\lambda u_s
\right]
$$

---

# 9. Model Characteristics

This optimization problem is a **Mixed-Integer Linear Program (MILP)** because:

- Binary variables $z_k$
- Linear objective
- Linear constraints

---

# 10. Expected Pyomo Implementation Structure

The Pyomo code should include:

### Sets

model.K
model.Kplus
model.L
model.S

### Parameters

model.D[k,s]
model.delta[k,s,l]
model.I0
model.P0[l]

### Variables

q[k]
z[k]

I[k,s]
P[k,s,l]

R[k,s]
lost[k,s]

u[s]

### Constraints
- ordering bounds
- arrivals
- lost sales
- inventory transition
- pipeline shift
- pipeline injection
- terminal deviation

### Objective
- minimize expected cost

---

# 11. Solver

The resulting model should be solved using a **MILP solver** (e.g., Gurobi, CPLEX, HiGHS, or CBC).

---

# 12. Implementation Notes

1. All scenario trajectories $D_{k,s}$ and $L_{k,s}$ must be generated **before building the model**.
2. The indicator $\delta_{k,s,\ell}$ should be precomputed.
3. The pipeline shift can be implemented directly in constraints without defining $\tilde P$.
4. Only $q_k$ and $z_k$ are scenario-independent decisions.
5. The model should be solved **once per MPC step**, and only $q_0$ executed.