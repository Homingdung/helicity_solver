# Three-regime helicity experiments on a prismatic cube.
#
# Single-shot initial-value runs (no time evolution, no mesh sweep).
# Each experiment combines:
#   * a domain (extruded unit cube, optionally periodic in z),
#   * an initial magnetic field tailored to the BC,
#   * the matching divergence-free projection + H(curl) helicity solve,
#   * an output .pvd for Paraview showing B, A, |B|, and a face tag.
#
# Cases:
#   closed       — B·n = 0 on all six faces;  H = ∫ A·B dV
#   line-tied    — B·n = B0·n on top/bottom only;  H_R = ∫ A·B dV
#   periodic     — z periodic, B·n = B0·n on lateral walls;
#                  H_gen = ∫ A·B dV  plus harmonic flux Φ_z = ⟨B_z⟩
#
# The same kernel works for all three:  ∫ A · B_func dV.  Differences
# live in (i) the init field, (ii) which faces carry Dirichlet data,
# (iii) whether the pressure carries a constant null space, (iv) whether
# a mixed-Poisson reference field B' is needed at all.

from firedrake import *
from firedrake.output import VTKFile
from functools import partial
import os

# Solver parameters (star-MG block, H(curl) helicity solve, AL Riesz
# preconditioner for the H(div)×L² saddle point) and the associated
# regularisation constants are imported from mesh_independence.py to
# keep both production scripts on a single tuned configuration.
from mesh_independence import (
    star_mg_params,
    spp_helicity,
    spp_riesz,
    beta_curl,
    gamma,
)

# ----------------------------------------------------------------------
# Resolution.  No time loop, so we can afford a richer mesh.  With
# N_base=8, extra_levels=2, N_layers=16 the finest mesh is 32×32×16
# prismatic → dofs(Vc) ≈ 270k.
# ----------------------------------------------------------------------
order = 1
N_base = 4           # horizontal cells per side on the base
extra_levels = 2     # MG refinements (finest = N_base * 2**extra_levels)
N_layers = 4        # vertical layers on the coarsest extruded mesh

dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}


def solve_constrained_div_free(Vd, Vn, bcs, options_prefix,
                               source=None, has_nullspace=True):
    """Saddle-point solve over Vd × Vn with the AL Riesz preconditioner.

    source=None  → mixed Poisson: B harmonic with B·n given by bcs.
    source given → L²-projection of `source` onto the div-free subspace
                   with B·n = bcs on Γ_D.
    """
    Z = MixedFunctionSpace([Vd, Vn])
    z = Function(Z)
    (B, p) = split(z)
    L = 0.5*inner(B, B)*dx - inner(p, div(B))*dx
    if source is not None:
        L = L - inner(source, B)*dx
    F = derivative(L, z, TestFunction(Z))

    U_aug = 0.5*(inner(B, B)
                 + inner(div(B)*gamma, div(B))
                 + inner(p*(1/gamma), p))*dx
    Jp = derivative(derivative(U_aug, z), z)

    nsp = None
    if has_nullspace:
        nsp = MixedVectorSpaceBasis(
            Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=Z.mesh().comm)])

    problem = NonlinearVariationalProblem(F, z, bcs=bcs, Jp=Jp)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=spp_riesz,
                                        options_prefix=options_prefix,
                                        nullspace=nsp,
                                        transpose_nullspace=nsp)
    solver.solve()
    its = solver.snes.getLinearSolveIterations()
    return Function(Vd).assign(z.subfunctions[0]), its


def solve_curl_curl(Vc, B_rhs, bcs, options_prefix, name="A"):
    """Singular curl-curl solve with β-regularised PC + residual lifting.

    (curl u, curl v) = (B_rhs, curl v), kernel = grad fields, handled by
    the β-PC + post-hoc Riesz lifting if MINRES leaves a non-trivial
    residual on the kernel.

    Returns (u_sol, its, lifted) where lifted is 1 if the residual
    lifting step ran, 0 otherwise.
    """
    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    u_sol = Function(Vc, name=name)
    a_cc  = inner(curl(u), curl(v))*dx
    L_cc  = inner(B_rhs,  curl(v))*dx
    Jp_cc = a_cc + inner(beta_curl*u, v)*dx

    prob = LinearVariationalProblem(a_cc, L_cc, u_sol, bcs=bcs, aP=Jp_cc)
    solver = LinearVariationalSolver(
        prob, solver_parameters=spp_helicity, options_prefix=options_prefix)

    def _riesz(functional, _s=solver):
        f = Function(functional.function_space().dual())
        with functional.dat.vec as x, f.dat.vec as y:
            _s.snes.ksp.pc.apply(x, y)
        return f

    solver.solve()
    its    = solver.snes.getLinearSolveIterations()
    lifted = solver.snes.ksp.getResidualNorm() > 0.01

    if lifted:
        r     = assemble(prob.F, bcs=prob.bcs)
        rstar = r.riesz_representation(riesz_map=_riesz, bcs=prob.bcs)
        c     = assemble(action(r, u_sol)) / assemble(action(r, rstar))
        out   = Function(Vc, name=name)
        out.assign(u_sol - c * rstar)
        return out, its, int(lifted)
    return u_sol, its, int(lifted)




# ----------------------------------------------------------------------
# Domain construction.
# ----------------------------------------------------------------------
def build_mesh(periodic):
    base = UnitSquareMesh(N_base, N_base, distribution_parameters=dparams)
    base_mh = MeshHierarchy(base, extra_levels,
                            distribution_parameters=dparams)
    builder = partial(ExtrudedMesh, periodic=True) if periodic else ExtrudedMesh
    mh = ExtrudedMeshHierarchy(base_mh, height=1.0,
                               base_layer=N_layers,
                               mesh_builder=builder)
    return mh[-1]


def build_function_spaces(mesh):
    """Return (Vc, Vd, Vn) on the prismatic extruded mesh."""
    h_n1c = FiniteElement("N1curl",   triangle, order)
    h_rt  = FiniteElement("RT",       triangle, order)
    h_cg  = FiniteElement("Lagrange", triangle, order)
    h_dg  = FiniteElement("DG",       triangle, order - 1)
    v_cg  = FiniteElement("Lagrange", interval, order)
    v_dg  = FiniteElement("DG",       interval, order - 1)

    Vc = FunctionSpace(mesh, HCurl(TensorProductElement(h_n1c, v_cg))
                             + HCurl(TensorProductElement(h_cg,  v_dg)))
    Vd = FunctionSpace(mesh, HDiv(TensorProductElement(h_rt,  v_dg))
                             + HDiv(TensorProductElement(h_dg,  v_cg)))
    Vn = FunctionSpace(mesh, TensorProductElement(h_dg, v_dg))
    return Vc, Vd, Vn


# ----------------------------------------------------------------------
# Per-regime configuration.
#
# Invariant across all three regimes:
#   * 4 lateral walls always carry  B·n = 0  (DirichletBC, homogeneous)
#   * A×n = 0 on every physical face we control
#
# The ONLY difference is what happens on top + bottom:
#   closed      : B·n = 0     (homogeneous Dirichlet)
#   line-tied   : B·n = B0·n  (inhomogeneous Dirichlet, "prescribed flux")
#   periodic    : top/bottom identified → no Dirichlet BC there
# ----------------------------------------------------------------------
def bc_config(bc_mode):
    """Return (periodic_z, needs_reference, flux_markers).

    flux_markers = boundary markers on which B·n is non-zero (used only
    for the ParaView `boundary_tag` field).  Lateral walls are *always*
    B·n = 0 so they never appear here.
    """
    if bc_mode == "closed":
        return False, False, ()
    if bc_mode == "line-tied":
        return False, True,  ("top", "bottom")
    if bc_mode == "periodic":
        return True,  False, ()
    raise ValueError(f"Unknown BC mode: {bc_mode!r}")


def bcs_for_B(W, B_init, bc_mode):
    """Build the DirichletBCs for B·n on the H(div) trace.

    Always: B·n = 0 on the 4 lateral walls (`"on_boundary"`).
    Top/bottom: 0 for closed, B_init·n for line-tied, omitted for periodic.
    """
    zero3 = Constant((0.0, 0.0, 0.0))
    bcs = [DirichletBC(W, zero3, "on_boundary")]
    if bc_mode == "closed":
        bcs += [DirichletBC(W, zero3,  "top"),
                DirichletBC(W, zero3,  "bottom")]
    elif bc_mode == "line-tied":
        bcs += [DirichletBC(W, B_init, "top"),
                DirichletBC(W, B_init, "bottom")]
    # periodic: top/bottom are identified, no physical BC there.
    return bcs


def bcs_for_A(Vc, bc_mode, variant="lateral"):
    """Build the DirichletBCs for A×n = 0 on the H(curl) trace.

    closed   :  A×n = 0 on every physical face.  H = ∫A·B is the
                standard closed-domain helicity in this gauge.
    periodic :  A×n = 0 on lateral walls only (top/bottom identified).
    line-tied:  variant-dependent.  All four variants are
                gauge-equivalent in the continuum: for line-tied,
                (B-B')·n = 0 on ∂Ω + div-free ⇒ ∫(B-B')dV = 0, which
                is precisely the Stokes consistency required for
                A×n = 0 on entire ∂Ω.  ablation_A_bcs.py confirms
                empirically that H_R agrees to machine precision
                across all four variants on the Finn-Antonsen test.
                The default "lateral" is the Berger-Field convention
                (A×n = 0 on the closed lateral sides, free on the
                open top/bottom footpoints).
            "lateral" : A×n = 0 on lateral walls only             [default]
            "full"    : A×n = 0 on entire ∂Ω
            "topbot"  : A×n = 0 on top/bottom only (lateral free)
            "none"    : no essential BC
    """
    zero3 = Constant((0.0, 0.0, 0.0))
    if bc_mode == "closed":
        return [DirichletBC(Vc, zero3, "on_boundary"),
                DirichletBC(Vc, zero3, "top"),
                DirichletBC(Vc, zero3, "bottom")]
    if bc_mode == "periodic":
        return [DirichletBC(Vc, zero3, "on_boundary")]
    if bc_mode == "line-tied":
        if variant == "lateral":
            return [DirichletBC(Vc, zero3, "on_boundary")]
        if variant == "full":
            return [DirichletBC(Vc, zero3, "on_boundary"),
                    DirichletBC(Vc, zero3, "top"),
                    DirichletBC(Vc, zero3, "bottom")]
        if variant == "topbot":
            return [DirichletBC(Vc, zero3, "top"),
                    DirichletBC(Vc, zero3, "bottom")]
        if variant == "none":
            return []
        raise ValueError(f"Unknown line-tied A variant: {variant!r}")
    raise ValueError(f"Unknown bc_mode: {bc_mode!r}")


def initial_field(bc_mode, X, Y, Z):
    """Return the analytical initial field B₀ for each regime."""
    if bc_mode == "closed":
        # B₀ = curl A₀ with A₀·t = 0 on every face, so B₀·n = 0
        # analytically.  The asymmetric frequencies (πx, 2πx) etc. break
        # the parity symmetry that would otherwise force ∫A·B = 0;
        # analytically ∫ A₀ · curl A₀ dV = −32/(3π²) ≈ −1.08.
        #   A₀ = ( sin(πy) sin(2πz),
        #          sin(2πx) sin(πz),
        #          sin(πx) sin(πy) )
        Bx = pi*sin(pi*X)*cos(pi*Y) - pi*sin(2*pi*X)*cos(pi*Z)
        By = 2*pi*sin(pi*Y)*cos(2*pi*Z) - pi*cos(pi*X)*sin(pi*Y)
        Bz = 2*pi*cos(2*pi*X)*sin(pi*Z) - pi*cos(pi*Y)*sin(2*pi*Z)
        return as_vector([Bx, By, Bz])

    if bc_mode == "line-tied":
        # Finn-Antonsen test configuration:
        #   reference :  B_p = ẑ,   A_p = (-y/2, x/2, 0)   (curl A_p = ẑ)
        #   bump      :  ψ(x,y,z) = α · x(1-x) · y(1-y) · z(1-z),
        #                ψ vanishes on the *entire* ∂Ω
        #   potential :  A = A_p + ψ ẑ
        #   field     :  B = curl A = (∂_y ψ, −∂_x ψ, 1)
        # Boundary structure:
        #   B·n = 0   on lateral walls  (∂_y ψ|_{x=0,1}=0, ∂_x ψ|_{y=0,1}=0)
        #   B·n = ±1  on top/bottom (uniform prescribed flux)
        # Relative helicity:
        #   H_R = ∫(A+A_p)·(B-B_p) dV = 2∫ψ dV = α·(1/6)³ · 2 = α/108
        #   α = 108 → H_R = 1.0
        alpha = Constant(108.0)
        Bx =  alpha *      X * (1 - X) * (1 - 2*Y) * Z * (1 - Z)   # = ∂_y ψ
        By = -alpha * (1 - 2*X) *      Y * (1 - Y) * Z * (1 - Z)   # = −∂_x ψ
        return as_vector([Bx, By, Constant(1.0)])

    if bc_mode == "periodic":
        # Vertical harmonic 1-form ê_z (constant z-mean) + a horizontal
        # swirl curl_z(-ψ ẑ) that vanishes identically on the 4 lateral
        # walls, so the "B·n = 0 on lateral walls" Dirichlet BC is
        # already satisfied by B_init — the L² projection then leaves
        # the field essentially unchanged.
        #
        #   ψ = sin(πx) sin(πy),    A = -ψ ẑ,    B_horiz = curl A
        #   B_x = -∂_y ψ = -π sin(πx) cos(πy)   (= 0 on x∈{0,1})
        #   B_y =  ∂_x ψ =  π cos(πx) sin(πy)   (= 0 on y∈{0,1})
        #   B_z = 1 (harmonic 1-form, not in image of curl on [0,1]²×S¹)
        # Analytical helicity  H = ∫ A·B = -∫ψ·1 dV = -(2/π)² = -4/π² ≈ -0.4053
        Bx = -pi*sin(pi*X)*cos(pi*Y)
        By =  pi*cos(pi*X)*sin(pi*Y)
        return as_vector([Bx, By, Constant(1.0)])

    raise ValueError(f"Unknown BC mode: {bc_mode!r}")


# ----------------------------------------------------------------------
# One experiment.
# ----------------------------------------------------------------------
def run_experiment(bc_mode, a_variant="lateral", outdir="output_helicity"):
    periodic, needs_reference, flux_markers = bc_config(bc_mode)
    mesh = build_mesh(periodic)
    Vc, Vd, Vn = build_function_spaces(mesh)

    X, Y, Z = SpatialCoordinate(mesh)
    B_init = initial_field(bc_mode, X, Y, Z)

    label = bc_mode if bc_mode != "line-tied" else f"line-tied_{a_variant}"

    # ------- Reference field B' (only line-tied needs one) -------------
    # closed   : B·n = 0 everywhere      → vacuum reference = 0
    # periodic : B·n = 0 on every face   → vacuum reference = 0
    if needs_reference:
        Z_ref = MixedFunctionSpace([Vd, Vn])
        bcs_ref = bcs_for_B(Z_ref.sub(0), B_init, bc_mode)
        B_prime, its_Bp = solve_constrained_div_free(
            Vd, Vn, bcs_ref,
            options_prefix=f"{label}_Bprime",
            has_nullspace=True)
        dofs_ref = Z_ref.dim()
    else:
        B_prime = Function(Vd, name="B_reference")
        dofs_ref = 0
        its_Bp = 0

    # ------- Divergence-free projection of B_init ----------------------
    Z_proj = MixedFunctionSpace([Vd, Vn])
    bcs_proj = bcs_for_B(Z_proj.sub(0), B_init, bc_mode)
    B_func, its_B = solve_constrained_div_free(
        Vd, Vn, bcs_proj,
        options_prefix=f"{label}_Bproj",
        source=B_init,
        has_nullspace=True)
    dofs_proj = Z_proj.dim()

    # ------- Companion vector potential A' for line-tied --------------
    # curl-curl singular solve with NO essential BC.  The RHS
    # (B', curl v) = ∮(B'×n)·v dS after IBP (curl B'=0); on lateral
    # walls (B'×n) is the only non-zero piece (B'=ẑ on top/bottom kills
    # it there).  Imposing A'×n=0 on lateral would force v×n=0 on the
    # only driving face, giving the trivial solution A'=0 — hence
    # bcs=[].  The β-term in the preconditioner selects the
    # L²-minimum-norm (Coulomb-like) gauge.
    if bc_mode == "line-tied":
        A_prime, its_Ap, lift_Ap = solve_curl_curl(
            Vc, B_prime, bcs=[],
            options_prefix=f"{label}_Aprime", name="A_prime")
        err_curlAp = float(norm(curl(A_prime) - B_prime, "L2"))
    else:
        A_prime = None
        its_Ap     = 0
        lift_Ap    = 0
        err_curlAp = 0.0

    # ------- H(curl) helicity solve: curl A = B_func - B_prime ---------
    # BC variant only matters for line-tied; the helpers default
    # closed/periodic to their canonical choices.  All four line-tied
    # variants are continuum gauge-equivalent (see ablation).
    B_field = B_func - B_prime
    A, its_A, lift_A = solve_curl_curl(
        Vc, B_field,
        bcs=bcs_for_A(Vc, bc_mode, variant=a_variant),
        options_prefix=f"{label}_helicity", name="A")

    err_curlA = float(norm(curl(A) - B_field, "L2"))

    # ------- Helicity integrals ---------------------------------------
    # Regime-specific definition (Table 2.1 of the thesis):
    #   closed     :  H    = ∫ A · B dV
    #   line-tied  :  H_R  = ∫ (A + A') · (B - B') dV       (Berger-Field)
    #   periodic   :  H̃    = ∫ A · (B + B_H) dV,
    #                 where  B_H = B − curl A  is the harmonic-1-form
    #                 residual (the ê_z mode on [0,1]² × S¹ that is
    #                 closed but not exact, so can never be matched by
    #                 curl A).
    if bc_mode == "periodic":
        B_H = B_func - curl(A)
        H = assemble(inner(A, B_func + B_H) * dx)
    elif bc_mode == "line-tied":
        # Finn–Antonsen relative helicity:
        #     H_R = ∫ (A_full + A') · (B - B') dV
        # In the code  curl(A) = B - B', so  A_full = A + A'  is the
        # potential of B, hence  H_R = ∫ (A + 2 A') · (B - B') dV.
        H = assemble(inner(A + 2.0 * A_prime, B_func - B_prime) * dx)
    else:
        H = assemble(inner(A, B_func) * dx)

    # Auxiliary readout for the periodic regime: harmonic flux ⟨B_z⟩.
    # On [0,1]² × S¹ this is the topological mode that ordinary
    # helicity cannot see directly.
    harmonic_flux = None
    if bc_mode == "periodic":
        harmonic_flux = assemble(B_func[2] * dx)  # volume = 1, so this = ⟨B_z⟩

    # ------- Paraview output -------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    Vvec_dg = VectorFunctionSpace(mesh, "DG", order - 1, dim=3)
    Vsca_dg = FunctionSpace(mesh, "DG", order - 1)

    B_out = Function(Vvec_dg, name="B").interpolate(B_func)
    A_out = Function(Vvec_dg, name="A").interpolate(A)
    Bref_out = Function(Vvec_dg, name="B_reference").interpolate(B_prime)
    Bmag = Function(Vsca_dg, name="B_magnitude").interpolate(sqrt(inner(B_func, B_func)))

    # Face tag: 1 where the face carries *prescribed flux* (B·n ≠ 0),
    # 0 elsewhere.  Lateral walls and homogeneous-Dirichlet faces both
    # read as 0; only the line-tied top/bottom light up.
    Vtag = FunctionSpace(mesh, "CG", 1)
    tag = Function(Vtag, name="boundary_tag")
    tag.assign(0.0)
    for m in flux_markers:
        DirichletBC(Vtag, 1.0, m).apply(tag)

    out_path = os.path.join(outdir, f"exp_{label}.pvd")
    VTKFile(out_path).write(B_out, A_out, Bref_out, Bmag, tag)

    dofs_A  = Vc.dim()
    dofs_Ap = Vc.dim() if A_prime is not None else 0

    if mesh.comm.rank == 0:
        print(f"  H={float(H):.6e}  err_curlA={err_curlA:.2e}  "
              f"iters(B'/B/A'/A)={its_Bp}/{its_B}/{its_Ap}/{its_A}  "
              f"dofs(A)={dofs_A}", flush=True)

    return {
        "bc_mode":        bc_mode,
        "a_variant":      a_variant if bc_mode == "line-tied" else "—",
        "label":          label,
        "H":              float(H),
        "harmonic_flux":  None if harmonic_flux is None else float(harmonic_flux),
        "err_curlA":      float(err_curlA),
        "err_curlAp":     float(err_curlAp),
        "its_Bp":         its_Bp,
        "its_B":          its_B,
        "its_Ap":         its_Ap,
        "its_A":          its_A,
        "lift_Ap":        lift_Ap,
        "lift_A":         lift_A,
        "dofs_A":         dofs_A,
        "dofs_ref":       dofs_ref,
        "dofs_proj":      dofs_proj,
        "dofs_Ap":        dofs_Ap,
        "pvd":            out_path,
    }


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def print_summary(results):
    """One unified summary table for the three regimes.

    Columns:
      regime, H, Phi_z, err_curlA, err_curlAp,
      its_Bp, its_B, its_Ap, its_A, dofs(A)
    Missing fields (Phi_z for non-periodic, A' columns for closed/periodic)
    print as `—`.
    """
    cols   = ("regime", "H", "Phi_z", "err_curlA", "err_curlAp",
              "its_Bp", "its_B", "its_Ap", "its_A", "dofs(A)")
    widths = (10, 16, 12, 11, 11, 7, 6, 7, 6, 9)
    header = "  ".join(f"{c:>{w}}" if i else f"{c:<{w}}"
                       for i, (c, w) in enumerate(zip(cols, widths)))
    total  = sum(widths) + 2 * (len(widths) - 1)

    print()
    print("=" * total)
    print("Helicity experiments — closed, line-tied, periodic")
    print("=" * total)
    print(header)
    print("-" * total)
    for r in results:
        phi    = "—" if r["harmonic_flux"] is None else f"{r['harmonic_flux']:.3e}"
        has_Ap = r["bc_mode"] == "line-tied"
        cAp    = f"{r['err_curlAp']:.2e}" if has_Ap else "—"
        iBp    = f"{r['its_Bp']}"          if has_Ap else "—"
        iAp    = f"{r['its_Ap']}"          if has_Ap else "—"
        print(f"{r['bc_mode']:<{widths[0]}}  "
              f"{r['H']:>{widths[1]}.6e}  "
              f"{phi:>{widths[2]}}  "
              f"{r['err_curlA']:>{widths[3]}.2e}  "
              f"{cAp:>{widths[4]}}  "
              f"{iBp:>{widths[5]}}  "
              f"{r['its_B']:>{widths[6]}}  "
              f"{iAp:>{widths[7]}}  "
              f"{r['its_A']:>{widths[8]}}  "
              f"{r['dofs_A']:>{widths[9]}}")


def write_results_file(path, results):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# Helicity experiments: closed + line-tied + periodic\n")
        f.write("# Columns: bc_mode  H  harmonic_flux  "
                "err_curlA  err_curlAp  its_Bp  its_B  its_Ap  its_A  "
                "lift_Ap  lift_A  dofs_A  dofs_ref  dofs_proj  dofs_Ap  "
                "pvd\n")
        for r in results:
            phi = "" if r["harmonic_flux"] is None else f"{r['harmonic_flux']:.10e}"
            f.write(f"{r['bc_mode']:<10} "
                    f"{r['H']:.10e}  {phi:<22} "
                    f"{r['err_curlA']:.6e}  {r['err_curlAp']:.6e}  "
                    f"{r['its_Bp']}  {r['its_B']}  {r['its_Ap']}  "
                    f"{r['its_A']}  {r['lift_Ap']}  {r['lift_A']}  "
                    f"{r['dofs_A']}  {r['dofs_ref']}  "
                    f"{r['dofs_proj']}  {r['dofs_Ap']}  {r['pvd']}\n")


if __name__ == "__main__":
    # One experiment per regime; line-tied uses the Berger-Field convention
    # (A×n = 0 on lateral walls only).  ablation_A_bcs.py covers the
    # gauge-invariance comparison of the four line-tied A-BC variants.
    experiments = [
        ("closed",    "lateral"),
        ("line-tied", "lateral"),
        ("periodic",  "lateral"),
    ]

    results = []
    for bc, variant in experiments:
        if COMM_WORLD.rank == 0:
            print(f"\n{'='*60}\nRunning experiment: {bc}\n{'='*60}",
                  flush=True)
        results.append(run_experiment(bc, a_variant=variant))

    if COMM_WORLD.rank == 0:
        print_summary(results)
        write_results_file("output_helicity/helicity_values.txt", results)
        print("\nWrote output_helicity/helicity_values.txt")
