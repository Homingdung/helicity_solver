# Mesh-independence test for the helicity solver.
#
# Domain: unit cube [0,1]^3, built as an extruded prismatic mesh
# (triangular base × interval).  The base hierarchy is a tet-free
# UnitSquareMesh + MeshHierarchy, extruded via ExtrudedMeshHierarchy
# so vertex-star multigrid sees a refinement chain in both the
# horizontal and vertical directions.
#
# Three boundary-condition regimes are supported (see ``BC_MODES``):
#   "closed"    – B·n = B_init·n on all 6 faces; A·t = 0 on all 6 faces.
#                 Lagrange multiplier p has a constant null space.
#   "line-tied" – B·n prescribed on top / bottom only; lateral walls carry
#                 the natural BC.  A·t = 0 only on top / bottom.  p is
#                 uniquely determined (no constant null space).
#   "periodic"  – z direction extruded periodically (no top / bottom
#                 boundary); B·n prescribed on the 4 lateral walls,
#                 A·t = 0 on the 4 lateral walls.  p has a constant null
#                 space.
#
# H(curl) helicity solve (Yang's solver): pure curl-curl problem
#     (curl u, curl v) = (B − B', curl v),  u ∈ H_0(curl, Γ_D),
# preconditioned by MINRES with an H(curl) Riesz preconditioner —
# vertex-star geometric multigrid (ASMStarPC, Firedrake
# hcurl_riesz_star demo) on the regularised operator
#     a(u,v) + β (u,v).
# After the MINRES solve a Riesz lifting using the residual functional
# is applied to remove the kernel contribution.
# Mixed H(div)×L² saddle point: MINRES + additive fieldsplit, with
# vertex-star multigrid on the H(div) block (hdiv_riesz_star demo) and
# Cholesky on the DG mass block.

from firedrake import *
from functools import partial
import csv
import os

# Boundary condition regimes to run.  Each entry produces its own set of
# CSV / LaTeX tables (suffixed with the regime name).  Set to a single
# regime (e.g. ["closed"]) to run only one experiment.
BC_MODES = ["closed", "line-tied", "periodic"]

order = 1
# Coarsest UnitSquareMesh size per direction.  Kept small (4) so each
# refinement adds one extra multigrid level on top of the base, giving
# vertex-star MG enough depth to saturate to its asymptotic contraction
# rate already on N=8.
N0 = 4

# Number of refinements applied to the base.  Finest = N0 * 2**k.
test_levels = [0, 1, 2]

dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

# Vertex-star multigrid relaxation block (firedrake.ASMStarPC) — used both
# as the H(curl) Riesz preconditioner for the helicity solve and as the
# (0,0)-block preconditioner inside the H(div)×L² fieldsplit.  Same pattern
# as the Firedrake demos:
#   https://www.firedrakeproject.org/demos/hcurl_riesz_star.py.html
#   https://www.firedrakeproject.org/demos/hdiv_riesz_star.py.html
# (HiptmairPC is not usable here because the prismatic H(curl) space is an
# EnrichedElement, for which Firedrake's dual basis is not implemented and
# the Hiptmair gradient interpolation therefore fails.)
star_mg_params = {
    "pc_type": "mg",
    "mg_levels": {
        "ksp_type":              "chebyshev",
        "ksp_max_it":            1,
        "pc_type":               "python",
        "pc_python_type":        "firedrake.ASMStarPC",
        "pc_star_construct_dim": 0,
        "pc_star_backend":       "tinyasm",
    },
    "mg_coarse": {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type":  "cholesky",
        "pc_factor_mat_solver_type": "mumps",
    },
}

# H(curl) helicity solve (Yang's solver): MINRES on the singular curl-curl
# operator, preconditioned (via aP) by an H(curl) Riesz preconditioner —
# one V-cycle of vertex-star geometric multigrid (ASMStarPC) applied to the
# regularised form  a(u,v) + β (u,v).  The MG is wrapped in AssembledPC so
# Firedrake's compute_operators sees a single PETSc-matrix handle for the
# preconditioning Jacobian.
spp_helicity = {
    "snes_type":             "ksponly",
    "ksp_type":              "minres",
    "ksp_max_it":            2000,
    "ksp_convergence_test":  "skip",
    "ksp_norm_type":         "preconditioned",
    "ksp_minres_nutol":      1.0e-12,
    # Diagnostic observability (printing only — no effect on the solve):
    # without these the singular curl-curl A'/A solves run completely
    # silently, so a non-converging grind toward ksp_max_it looks like a
    # hang.  Shows the MINRES residual history + why the solve stopped.
    "ksp_monitor":           None,
    "ksp_converged_reason":  None,
    "mat_type":              "aij",
    "pc_type":               "python",
    "pc_python_type":        "firedrake.AssembledPC",
    "assembled":             star_mg_params,
}
# Regularisation weight for the curl-curl preconditioner.
beta_curl = Constant(0.1)

# Augmented-Lagrangian Riesz preconditioner for the H(div)×L² saddle point.
gamma = Constant(1.0e5)
spp_riesz = {
    "mat_type": "nest",
    "snes_type": "ksponly",
    "snes_monitor": None,
    "ksp_monitor": None,
    "ksp_max_it": 1000,
    "ksp_norm_type": "preconditioned",
    "ksp_type": "minres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled": star_mg_params,
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    "ksp_atol": 1.0e-50,
    "ksp_rtol": 1.0e-12,
}

def solve_constrained_div_free(Vd, Vn, bcs, options_prefix, source=None,
                               has_nullspace=True):
    """Saddle-point solve over Vd × Vn with the AL Riesz preconditioner.

    Common Lagrangian:
        L(B, p) = 1/2 ⟨B, B⟩ − ⟨source, B⟩ − ⟨p, div B⟩.

    ``source=None`` → mixed Poisson: B harmonic with B·n given by ``bcs``.
    ``source`` provided → L²-projection of ``source`` onto
        { B ∈ Vd : div B = 0, B·n = bcs on Γ_D }.

    ``has_nullspace=True`` declares the constant null space for the
    Lagrange multiplier p (needed when B·n is prescribed over the entire
    boundary, e.g. ``closed`` / ``periodic``).  When part of the boundary
    carries the natural BC (``line-tied``), p is uniquely determined and
    no null space is declared.

    Returns ``(B_func, linear_its)``.
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


def _bc_config(bc_mode):
    """Return ``(periodic, side_markers, has_p_nullspace)`` for a BC regime."""
    if bc_mode == "closed":
        return False, ("on_boundary", "top", "bottom"), True
    if bc_mode == "line-tied":
        return False, ("top", "bottom"),                False
    if bc_mode == "periodic":
        return True,  ("on_boundary",),                 True
    raise ValueError(f"Unknown BC mode: {bc_mode!r} "
                     f"(expected one of 'closed', 'line-tied', 'periodic')")


def run_mesh_independence(bc_mode):
    """Run the three-stage mesh-independence sweep for one BC regime.

    Returns lists of result rows for (helicity, mixed, projection).  The
    helicity rows include the potential-check error ``||B - curl A||_L2``
    as a fifth column.
    """
    periodic, side_markers, has_p_nullspace = _bc_config(bc_mode)

    csv_path = f"output/mesh_independence_{bc_mode}.csv"
    csv_header = ["N", "h",
                  "dofs_R",  "iter_R",         # Algorithm 1, Mode R (ref. field B')
                  "dofs_P",  "iter_P",         # Algorithm 1, Mode P (projection)
                  "iter_Ap", "err_curlAp",     # Algorithm 2 on B' (line-tied only)
                  "dofs_A",  "iter_A",         # Algorithm 2 (vector potential A)
                  "err_curlA"]

    if COMM_WORLD.rank == 0:
        os.makedirs("output", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    def _append_csv(row):
        if COMM_WORLD.rank != 0:
            return
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    results_helicity = []
    results_mixed = []
    results_projection = []

    if COMM_WORLD.rank == 0:
        print(f"\n############################################################")
        print(f"#  BC mode: {bc_mode}  "
              f"(periodic_z={periodic}, side_markers={side_markers}, "
              f"p_nullspace={has_p_nullspace})")
        print(f"############################################################",
              flush=True)

    for k in test_levels:
        base    = UnitSquareMesh(N0, N0, distribution_parameters=dparams)
        base_mh = MeshHierarchy(base, k, distribution_parameters=dparams)
        builder = partial(ExtrudedMesh, periodic=True) if periodic else ExtrudedMesh
        mh      = ExtrudedMeshHierarchy(base_mh, height=1.0,
                                        base_layer=N0, mesh_builder=builder)
        mesh    = mh[-1]

        # Function spaces on triangular prisms.
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

        (X0, Y0, Z0) = SpatialCoordinate(mesh)
        # 2π in z so the test field is periodic across the top/bottom
        # interface when ``periodic`` is enabled.
        B_init = as_vector([sin(2*pi*X0)*sin(2*pi*Y0),
                            cos(2*pi*X0)*cos(2*pi*Y0),
                            1.0 + 0.5*sin(2*pi*Z0)])

        def _bcs(W, value):
            return [DirichletBC(W, value, m) for m in side_markers]

        # ---------------- Reference field B' (mixed Poisson) -------------
        Z_ref = MixedFunctionSpace([Vd, Vn])
        bcs_ref = _bcs(Z_ref.sub(0), B_init)
        B_prime, its_mixed = solve_constrained_div_free(
            Vd, Vn, bcs_ref, options_prefix=f"{bc_mode}_B_prime_ref",
            has_nullspace=has_p_nullspace)

        # ---------------- Divergence-free projection of B_init -----------
        Z_proj = MixedFunctionSpace([Vd, Vn])
        bcs_proj = _bcs(Z_proj.sub(0), B_init)
        B_func, its_proj = solve_constrained_div_free(
            Vd, Vn, bcs_proj,
            options_prefix=f"{bc_mode}_B_init_div_free_projection",
            source=B_init,
            has_nullspace=has_p_nullspace)

        # ---------------- Companion potential A' for line-tied ------------
        # Singular curl-curl for the reference field B':
        #     (curl u, curl v) = (B', curl v),  bcs = []
        # Same β-regularised PC and residual lifting as the main A solve.
        # The driving boundary contribution lives on the lateral walls
        # (where B' is non-trivial); imposing A'×n = 0 there would kill
        # the RHS, so we leave A' fully free.
        if bc_mode == "line-tied":
            uAp = TrialFunction(Vc)
            vAp = TestFunction(Vc)
            Ap_sol = Function(Vc, name="A_prime")
            a_ap  = inner(curl(uAp), curl(vAp))*dx
            L_ap  = inner(B_prime,   curl(vAp))*dx
            Jp_ap = a_ap + inner(beta_curl*uAp, vAp)*dx
            prob_ap = LinearVariationalProblem(a_ap, L_ap, Ap_sol,
                                               bcs=[], aP=Jp_ap)
            ap_solver = LinearVariationalSolver(
                prob_ap, solver_parameters=spp_helicity,
                options_prefix=f"{bc_mode}_Aprime")
            ap_solver.solve()
            its_Ap     = ap_solver.snes.getLinearSolveIterations()
            err_curlAp = float(norm(curl(Ap_sol) - B_prime, "L2"))
        else:
            its_Ap     = 0
            err_curlAp = 0.0

        # ---------------- H(curl) helicity solve --------------------------
        B_field = B_func - B_prime
        u = TrialFunction(Vc)
        v = TestFunction(Vc)
        u_sol = Function(Vc)
        a_curl  = inner(curl(u), curl(v))*dx
        L_curl  = inner(B_field, curl(v))*dx
        Jp_curl = a_curl + inner(beta_curl*u, v)*dx
        bcs_curl = _bcs(Vc, 0)

        problem = LinearVariationalProblem(a_curl, L_curl, u_sol,
                                           bcs=bcs_curl, aP=Jp_curl)
        helicity_solver = LinearVariationalSolver(
            problem, solver_parameters=spp_helicity,
            options_prefix=f"{bc_mode}_helicity")

        def riesz_map(functional, _solver=helicity_solver):
            function = Function(functional.function_space().dual())
            with functional.dat.vec as x, function.dat.vec as y:
                _solver.snes.ksp.pc.apply(x, y)
            return function

        helicity_solver.solve()
        its_helicity = helicity_solver.snes.getLinearSolveIterations()

        # Residual lifting: project out the kernel mode that MINRES
        # cannot resolve on the singular curl-curl operator.
        if helicity_solver.snes.ksp.getResidualNorm() > 0.01:
            r = assemble(problem.F, bcs=problem.bcs)
            rstar = r.riesz_representation(riesz_map=riesz_map, bcs=problem.bcs)
            c = assemble(action(r, u_sol)) / assemble(action(r, rstar))
            A = Function(Vc, name="MagneticPotential")
            A.assign(u_sol - c * rstar)
        else:
            A = u_sol

        err_curlA = norm(curl(A) - B_field, "L2")
        N_finest  = N0 * 2**k
        # Total DOF per problem: mixed problems use Z.dim() of their
        # MixedFunctionSpace; helicity is a single-space H(curl) solve.
        dofs_ref  = Z_ref.dim()
        dofs_proj = Z_proj.dim()
        dofs_A    = Vc.dim()

        target_label = "B* - curl(A)" if bc_mode == "line-tied" else "B - curl(A)"
        if COMM_WORLD.rank == 0:
            ap_str = (f", its(Ap)={its_Ap}, ||B'-curl(A')||_L2={err_curlAp:.6e}"
                      if bc_mode == "line-tied" else "")
            print(f"[done] bc={bc_mode}, k={k}, N={N_finest}, "
                  f"dofs(A)={dofs_A}, dofs(ref)={dofs_ref}, dofs(proj)={dofs_proj}, "
                  f"its(helicity)={its_helicity}, its(mixed)={its_mixed}, "
                  f"its(projection)={its_proj}{ap_str}, "
                  f"||{target_label}||_L2={err_curlA:.6e}", flush=True)

        results_helicity.append([N_finest, dofs_A, its_helicity, err_curlA,
                                 its_Ap, err_curlAp])
        results_mixed.append(    [N_finest, dofs_ref,  its_mixed])
        results_projection.append([N_finest, dofs_proj, its_proj])

        _append_csv([N_finest, 1.0/N_finest,
                     dofs_ref,  its_mixed,
                     dofs_proj, its_proj,
                     its_Ap,    err_curlAp,
                     dofs_A,    its_helicity,
                     err_curlA])

    return results_helicity, results_mixed, results_projection


# ============================================================
# Run every requested BC regime in turn.
# ============================================================
if __name__ == "__main__":
    all_results = {}
    for bc in BC_MODES:
        all_results[bc] = run_mesh_independence(bc)

# ============================================================
# Output: LaTeX tables to stdout (CSVs are written incrementally above).
# ============================================================
if __name__ == "__main__" and COMM_WORLD.rank == 0:
    # Notation per BC regime.  In all regimes we numerically verify the
    # same quantity, ``B_func - B_prime - curl A``.  In closed / periodic
    # we follow the convention that ``B`` already denotes the
    # zero-normal-trace closed field (so the reference is absorbed into
    # B); in line-tied we keep the footpoint reference explicit as
    # ``B* := B - B'``.
    def _helicity_target(bc):
        return r"B^{\star}" if bc == "line-tied" else r"B"

    def _err_tex(eh):
        mant, expn = f"{eh:.1e}".split("e")
        return rf"${mant} \times 10^{{{int(expn)}}}$"

    def _header_cp(target):
        # closed + periodic: B', B, A — no A' stage.
        return [
            r"\begin{tabular}{cc|cc|cc|ccc}",
            r"\toprule",
            (r"BC & "
             r"& \multicolumn{2}{c|}{Algorithm 1, Mode R} "
             r"& \multicolumn{2}{c|}{Algorithm 1, Mode P} "
             r"& \multicolumn{3}{c}{Algorithm 2} \\"),
            (r" & "
             r"& \multicolumn{2}{c|}{(ref.\ field $B'$, $s=0$)} "
             r"& \multicolumn{2}{c|}{(projection of $B_0$, $s=B_0$)} "
             r"& \multicolumn{3}{c}{(vector potential $A$, curl-curl)} \\"),
            (rf"regime & $h$ & \#DoFs & Iter & \#DoFs & Iter & \#DoFs & Iter "
             rf"& $\|{target} - \mathrm{{curl}}\,A\|_{{L^2}}$ \\"),
            r"\midrule",
        ]

    def _header_lt(target):
        # line-tied: add Iter and residual for A' between Mode P and the
        # main A block.  A' lives in the same Vc as A, so its #DoFs is
        # identical and we suppress it.
        return [
            r"\begin{tabular}{cc|cc|cc|cc|ccc}",
            r"\toprule",
            (r"BC & "
             r"& \multicolumn{2}{c|}{Algorithm 1, Mode R} "
             r"& \multicolumn{2}{c|}{Algorithm 1, Mode P} "
             r"& \multicolumn{2}{c|}{Algorithm 2} "
             r"& \multicolumn{3}{c}{Algorithm 2} \\"),
            (r" & "
             r"& \multicolumn{2}{c|}{(ref.\ field $B'$, $s=0$)} "
             r"& \multicolumn{2}{c|}{(projection of $B_0$, $s=B_0$)} "
             r"& \multicolumn{2}{c|}{(companion $A'$, curl-curl)} "
             r"& \multicolumn{3}{c}{(vector potential $A$, curl-curl)} \\"),
            (rf"regime & $h$ & \#DoFs & Iter & \#DoFs & Iter "
             rf"& Iter & $\|B' - \mathrm{{curl}}\,A'\|_{{L^2}}$ "
             rf"& \#DoFs & Iter "
             rf"& $\|{target} - \mathrm{{curl}}\,A\|_{{L^2}}$ \\"),
            r"\midrule",
        ]

    def _row_lines_cp(bc, rows_h, rows_m, rows_p):
        lines = []
        for i, (rh, (_, dm, im), (_, dp, ip)) in \
                enumerate(zip(rows_h, rows_m, rows_p)):
            Nh, dh, ih, eh = rh[:4]
            tag = bc if i == 0 else ""
            lines.append(
                rf"{tag} & $1/{Nh}$ & {dm} & {im} & {dp} & {ip} & "
                rf"{dh} & {ih} & {_err_tex(eh)} \\")
        return lines

    def _row_lines_lt(bc, rows_h, rows_m, rows_p):
        lines = []
        for i, (rh, (_, dm, im), (_, dp, ip)) in \
                enumerate(zip(rows_h, rows_m, rows_p)):
            Nh, dh, ih, eh, iap, eap = rh
            tag = bc if i == 0 else ""
            lines.append(
                rf"{tag} & $1/{Nh}$ & {dm} & {im} & {dp} & {ip} & "
                rf"{iap} & {_err_tex(eap)} & "
                rf"{dh} & {ih} & {_err_tex(eh)} \\")
        return lines

    def _emit_combined(bcs, target, header_fn, row_fn):
        lines = header_fn(target)
        for j, bc in enumerate(bcs):
            rows_h, rows_m, rows_p = all_results[bc]
            if j > 0:
                lines.append(r"\midrule")
            lines.extend(row_fn(bc, rows_h, rows_m, rows_p))
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    cp_bcs = [bc for bc in BC_MODES if bc in ("closed", "periodic")]
    lt_bcs = [bc for bc in BC_MODES if bc == "line-tied"]

    if cp_bcs:
        print("\n" + "#" * 60)
        print("#  Table 1 — closed + periodic")
        print("#" * 60)
        print("\n=== KSP iteration counts and helicity residual "
              "(closed + periodic) ===\n")
        print(_emit_combined(cp_bcs, _helicity_target("closed"),
                             _header_cp, _row_lines_cp))

    if lt_bcs:
        print("\n" + "#" * 60)
        print("#  Table 2 — line-tied")
        print("#" * 60)
        print("\n=== KSP iteration counts and helicity residual "
              "(line-tied) ===\n")
        print(_emit_combined(lt_bcs, _helicity_target("line-tied"),
                             _header_lt, _row_lines_lt))

    for bc in BC_MODES:
        print(f"\nSaved CSV to output/mesh_independence_{bc}.csv")
