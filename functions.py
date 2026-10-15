#===========================
# IMPORTING APPROPRIATE LIBRARIES AND MODULES:
#===========================
 
import numpy as np
import math
 
 
 
#===========================
# BISECTION ROOT FINDER:
#===========================

def bisection_root(f, lo, hi, tol=1e-13, max_iter=300):
    """Root of f in [lo, hi]; f(lo) and f(hi) must differ in sign."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("bisection_root: no sign change on [%g, %g]" % (lo, hi))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0.0 or 0.5 * (hi - lo) < tol * max(1.0, abs(mid)):
            return mid
        if flo * fmid < 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def all_roots(f, grid):
    """All roots of f on a grid, found by scanning for sign changes + bisection."""
    roots, vals = [], [f(g) for g in grid]
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            roots.append(bisection_root(f, grid[i], grid[i + 1]))
    return roots


#===========================
# QUADRATIC SOLVER:
#===========================

def solve_dvdx_quadratic(m1, m2, m3):
    """Return two real roots, or two NaNs if no real roots exist."""

    if m1 == 0:
        if m2 == 0:
            return np.nan, np.nan

        # Linear equation: one root, no second quadratic root
        return -m3 / m2, np.nan

    discriminant = m2**2 - 4*m1*m3

    if discriminant < 0:
        return np.nan, np.nan

    sqrt_D = math.sqrt(discriminant)

    root_1 = (-m2 + sqrt_D) / (2*m1)
    root_2 = (-m2 - sqrt_D) / (2*m1)

    return root_1, root_2


#===========================
# CRITICAL MACH NUMBER FINDER:
#===========================

def m_xc(p):
  gamma=p.gamma
  alpha_pi=p.alpha_pi
  I_n=p.I_n
  g=p.g
  m1 = (gamma**2) * ((alpha_pi**2) * I_n * (2*g - 1) * (gamma - 1) - (gamma + 1))
  m2 = 2 * gamma * (gamma + (alpha_pi**2) * I_n * g * (g - 1) * (gamma - 1))
  m3 = -(alpha_pi**2) * I_n * (g**2) * (gamma - 1)
  discriminant = (m2**2) - (4 * m1 * m3)
  if discriminant < 0:
        raise ValueError("Discriminant (m2^2 - 4*m1*m3) is negative, resulting in complex numbers.")
  inner_sqrt = math.sqrt(discriminant)
  fraction = (-m2 - inner_sqrt) / (2 * m1)
  if fraction < 0:
        raise ValueError("Fraction inside the outer square root is negative, resulting in complex numbers.")
  M_xc = math.sqrt(fraction)
  return M_xc, m1, m2, m3

#===========================
# CRITICAL SPEED OF SOUND FUNCTION:
#===========================

def a_xc_equation(a_x,p,x_in,lambda_in):
    gamma = p.gamma
    alpha_pi = p.alpha_pi
    I_n = p.I_n
    g = p.g
    S = p.S

    M_xc, m1, m2, m3 = m_xc(p)
    M = M_xc

    A = S / ((x_in**1.5) * (x_in - 1))
    B_term1 = (alpha_pi**2 * I_n * (g + gamma * M**2)**2) / (gamma**2 * x_in)
    B_term2 = (alpha_pi**2 * I_n * g * (5*x_in - 3) * (g + gamma * M**2)) / (gamma**2 * x_in * (x_in - 1))
    B_term3 = (M**2 * (5*x_in - 3)) / (x_in * (gamma - 1) * (x_in - 1))
    B = B_term1 + B_term2 - B_term3
    C = -(2 * lambda_in * alpha_pi * I_n * M * (g + gamma * M**2)) / (gamma * x_in**2)
    D_part1 = (lambda_in**2 / x_in**3) - (1 / (2 * (x_in - 1)**2))
    D_part2 = ((gamma + 1) * M**2) / (gamma - 1) - (2 * alpha_pi**2 * g * I_n * (g + gamma * M**2)) / gamma
    D = -(D_part1 * D_part2)
    F = A * (a_x**3) + B * (a_x**2) + C * a_x + D
    return F

#===========================
# N AND D FINDER:
#===========================

def ND_finder(p,x_in,a,v,lambda_in):
  
    gamma = p.gamma
    alpha_pi = p.alpha_pi
    I_n = p.I_n
    g = p.g
    S = p.S

 
    # Helper sub-expression common to both equations
    ga2_gv2 = g * a**2 + gamma * v**2

    
    # NUMERATOR N (Equation 10a)
    # Variables: gamma, alpha_pi, I_n, g, S, x_in, lambda_in, v, a
    
    N = (
        ((S * a**5) / (x_in**1.5 * (x_in - 1.0)))
        + (alpha_pi**2 * I_n * ga2_gv2**2) / (gamma**2 * x_in)
        + (alpha_pi**2 * g * I_n * a**2 * (5.0 * x_in - 3.0) * ga2_gv2) / (gamma**2 * x_in * (x_in - 1.0))
        - ((lambda_in**2 / x_in**3) - (1.0 / (2.0 * (x_in - 1.0)**2))) * (
            ((gamma + 1.0) * v**2 / (gamma - 1.0)) - (2.0 * alpha_pi**2 * g * I_n * ga2_gv2 / gamma)
        )
        - (v**2 * a**2 * (5.0 * x_in - 3.0)) / (x_in * (gamma - 1.0) * (x_in - 1.0))
        - (2.0 * lambda_in * alpha_pi * I_n * v * ga2_gv2) / (gamma * x_in**2)
    )

  
    # DENOMINATOR D (Equation 10b)
    # Variables: gamma, alpha_pi, I_n, g, S, x_in, lambda_in, v, a
    
    D = (
        (2.0 * a**2 * v) / (gamma - 1.0)
        - ((gamma + 1.0) * v**3) / (gamma - 1.0)
        + (alpha_pi**2 * I_n * v * ga2_gv2 / gamma) * ((2.0 * g - 1.0) - (g * a**2) / (gamma * v**2))
    )
    
    return N,D

#===========================
# DV/DX QUADRATIC FUNCTION:
#===========================

def dvdx_equation(dvdx,p,a,v,x_in,lambda_in):
    gamma = p.gamma
    alpha_pi = p.alpha_pi
    I_n = p.I_n
    g = p.g
    S = p.S

    # PARTIAL DERIVATIVES OF NUMERATOR N(x_in, v, a, lambda_in)

    # 1. Partial derivative with respect to velocity v (N_v)
    N_v = (
        (4.0 * alpha_pi**2 * I_n * v * (g * a**2 + gamma * v**2)) / (gamma * x_in)
        + (2.0 * alpha_pi**2 * g * I_n * a**2 * v * (5.0 * x_in - 3.0)) / (gamma * x_in * (x_in - 1.0))
        - ((lambda_in**2 / x_in**3) - (1.0 / (2.0 * (x_in - 1.0)**2))) * (((2.0 * (gamma + 1.0) * v) / (gamma - 1.0)) - 4.0 * alpha_pi**2 * g * I_n * v)
        - (2.0 * v * a**2 * (5.0 * x_in - 3.0)) / (x_in * (gamma - 1.0) * (x_in - 1.0))
        - (2.0 * lambda_in * alpha_pi * I_n * (g * a**2 + 3.0 * gamma * v**2)) / (gamma * x_in**2)
    )

    # 2. Partial derivative with respect to sound speed a (N_a)
    N_a = (
        (5.0 * S * a**4) / (x_in**1.5 * (x_in - 1.0))
        + (4.0 * alpha_pi**2 * g * I_n * a * (g * a**2 + gamma * v**2)) / (gamma**2 * x_in)
        + (2.0 * alpha_pi**2 * g * I_n * a * (5.0 * x_in - 3.0) * (2.0 * g * a**2 + gamma * v**2)) / (gamma**2 * x_in * (x_in - 1.0))
        + ((lambda_in**2 / x_in**3) - (1.0 / (2.0 * (x_in - 1.0)**2))) * (4.0 * alpha_pi**2 * g**2 * I_n * a / gamma)
        - (2.0 * v**2 * a * (5.0 * x_in - 3.0)) / (x_in * (gamma - 1.0) * (x_in - 1.0))
        - (4.0 * lambda_in * alpha_pi * g * I_n * v * a) / (gamma * x_in**2)
    )

    # 3. Partial derivative with respect to angular momentum lambda_in (N_lambda)
    N_lambda = (
        - (2.0 * lambda_in / x_in**3) * (((gamma + 1.0) * v**2 / (gamma - 1.0)) - (2.0 * alpha_pi**2 * g * I_n * (g * a**2 + gamma * v**2) / gamma))
        - (2.0 * alpha_pi * I_n * v * (g * a**2 + gamma * v**2)) / (gamma * x_in**2)
    )

    # 4. Partial derivative with respect to radial distance x_in (N_x)
    N_x = (
        - (S * a**5 * (5.0 * x_in - 3.0)) / (2.0 * x_in**2.5 * (x_in - 1.0)**2)
        - (alpha_pi**2 * I_n * (g * a**2 + gamma * v**2)**2) / (gamma**2 * x_in**2)
        + (alpha_pi**2 * g * I_n * a**2 * (g * a**2 + gamma * v**2) / gamma**2) * ((-5.0 * x_in**2 + 6.0 * x_in - 3.0) / (x_in**2 * (x_in - 1.0)**2))
        - ((-3.0 * lambda_in**2 / x_in**4) + (1.0 / (x_in - 1.0)**3)) * (((gamma + 1.0) * v**2 / (gamma - 1.0)) - (2.0 * alpha_pi**2 * g))
    )

    # PARTIAL DERIVATIVES OF DENOMINATOR D(x_in, v, a, lambda_in)

    # 1. Partial derivative with respect to velocity v (D_v)
    D_v = (
        (2.0 * a**2 - 3.0 * (gamma + 1.0) * v**2) / (gamma - 1.0)
        + 3.0 * alpha_pi**2 * I_n * (2.0 * g - 1.0) * v**2
        + (2.0 * alpha_pi**2 * I_n * g * (g - 1.0) * a**2) / gamma
        + (alpha_pi**2 * I_n * g**2 * a**4) / (gamma**2 * v**2)
    )

    # 2. Partial derivative with respect to sound speed a (D_a)
    D_a = (
        (4.0 * a * v) / (gamma - 1.0)
        + (4.0 * alpha_pi**2 * I_n * g * (g - 1.0) * a * v) / gamma
        - (4.0 * alpha_pi**2 * I_n * g**2 * a**3) / (gamma**2 * v)
    )

    # COEFFICIENTS OF da/dx FROM EQUATION (11)
    # da_dx = A1 * dv_dx + A2
    # Variables: gamma, alpha_pi, I_n, g, S, x_in, lambda_in, v, a

    # 1. Coefficient of dv_dx (A1)
    A1 = (a / v) - (gamma * v / a)

    # 2. Remaining RHS terms independent of dv_dx (A2)
    A2 = ((gamma / a) * ((lambda_in**2 / x_in**3) - (1.0 / (2.0 * (x_in - 1.0)**2))) + ((5.0 * x_in - 3.0) * a) / (2.0 * x_in * (x_in - 1.0)))

    # COEFFICIENTS OF dlambda/dx FROM EQUATION (12)
    # dlambda_dx = B1 * dv_dx + B2 * da_dx + B3
    # Variables: gamma, alpha_pi, I_n, g, S, x_in, lambda_in, v, a

    # 1. Coefficient of dv_dx (B1)
    B1 = -(alpha_pi * x_in * (g * a**2 - gamma * v**2)) / (gamma * v**2)

    # 2. Coefficient of da_dx (B2)
    B2 = (2.0 * alpha_pi * a * x_in * g) / (gamma * v)

    # 3. Remaining RHS terms independent of derivatives (B3)
    B3 = (alpha_pi * (g * a**2 + gamma * v**2)) / (gamma * v)

    m1 = D_v + D_a*A1
    m2 = D_a*A2 - N_v - N_a*A1 - N_lambda*(A1*B2 + B1)
    m3 = N_a*A2 - N_lambda*(A2*B2 + B3) - N_x

    F = m1*(dvdx**2) + m2*(dvdx) + m3

    return F,m1,m2,m3

#===========================
# ENERGY AT CRITICAL POINT FINDER:
#===========================

def energy(p,a,v,x_in,lambda_in):
    gamma = p.gamma

    epsilon = (v**2) / 2 + (a**2) / (gamma - 1) + (lambda_in**2) / (2 * x_in**2) + (-1/2) * ((x_in - 1)**(-1))

    return epsilon

#===========================
# MASTER FUNCTION TO FIND ALL CRITICAL POINT PARAMETERS:
#===========================

def master_critical_point_solver(p, x_in, lambda_in):
    """
    Master function to calculate M_xc, a_xc, v_xc, dv/dx roots, and energy.
    Assumes m_xc, a_xc_equation, dvdx_equation, energy, and all_roots are defined.
    """

    # 1. Calculate Critical Mach Number (M_xc)
    M_xc = m_xc(p)[0]


    # 2. Calculate sound speed at the critical point
    f_target = lambda a_x: a_xc_equation(
        a_x, p, x_in, lambda_in
    )

    search_grid = np.linspace(0.001, 10, 1000)
    found_roots = all_roots(f_target, search_grid)

    # Keep only positive roots, ALTHOUGH THE SEARCH GRID ALREADY ENSURES POSITIVE ROOTS, THIS IS A SAFETY CHECK
    positive_roots = [root for root in found_roots if root > 0]

    if not positive_roots:
        raise ValueError(
            f"No positive sound-speed root detected for "
            f"x_in={x_in:.4f}, lambda_in={lambda_in:.3f}"
        )

    a_xc_val = positive_roots[0]


    # 3. Calculate Speed of Accreting Gas at Critical Point (v_xc)
    v_xc = M_xc * a_xc_val

    # 4. Calculate both roots of the velocity gradient (dv/dx)
    # Lock in the parameter object, the computed a_xc and v_xc, and the new inputs
    _, m1, m2, m3 = dvdx_equation(0, p, a_xc_val, v_xc, x_in, lambda_in)

    dvdx_root_1, dvdx_root_2 = solve_dvdx_quadratic(m1, m2, m3)

    dvdx_roots = (dvdx_root_1, dvdx_root_2)

    # 5. Calculate specific energy
    eps = energy(p, a_xc_val, v_xc, x_in, lambda_in)

    # 6. N D values for the critical point
    N, D = ND_finder(p, x_in, a_xc_val, v_xc, lambda_in)

    #dvdx equation discriminant
    discriminant = (m2**2) - (4 * m1 * m3)

    return M_xc, v_xc, a_xc_val, N, D, dvdx_roots, eps, discriminant


