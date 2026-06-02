# Helicity convergence study on the prismatic unit cube.
#
# Built on top of helicity_experiments.py.  Instead of a single-shot run
# at one fixed resolution, this script sweeps the mesh-refinement level
# (default 1, 2, 3, 4) for each boundary-condition regime and compares the
# *computed* helicity against the known *analytic* helicity, so we obtain
#
#       level → (DOFs, KSP iterations, error, rate of convergence)
#
# for every regime.  The analytic helicity of each test field is fixed by
# construction (see helicity_experiments.initial_field):
#
#   closed     :  H   = ∫ A₀ · curl A₀ dV = −32 / (3 π²)  ≈ −1.0808
#   line-tied  :  H_R = α / 108 = 1.0           (α = 108, Finn–Antonsen)
#   periodic   :  H̃  = ∫ A·(B + B_H) dV = −2∫ψ = −8/π² ≈ −0.8106
#                 (B_H = B − curl A ≈ ê_z is the harmonic-1-form residual;
#                  the ê_z mode contributes a second −∫ψ on top of ∫A·B)
#
# With h_k = 1 / (N_base · 2^k) the mesh size halves every level, so the
# observed convergence rate between consecutive levels is
#
#       rate_k = log2( e_{k-1} / e_k ).
#
# This script does NOT write ParaView output — it is a pure convergence
# harness.  Run it on a compute node (SSH); the level-4 meshes are large.
#
#   Examples:
#     python helicity_convergence_study.py
#     python helicity_convergence_study.py --regimes closed periodic
#     python helicity_convergence_study.py --levels 1 2 3 --n-base 2
#     mpiexec -n 8 python helicity_convergence_study.py

import argparse
import csv
import datetime
import math
import os
import subprocess
from functools import partial

from firedrake import (
    COMM_WORLD,
    DistributedMeshOverlapType,
    ExtrudedMesh,
    ExtrudedMeshHierarchy,
    Function,
    MeshHierarchy,
    MixedFunctionSpace,
    SpatialCoordinate,
    UnitSquareMesh,
    assemble,
    curl,
    dx,
    inner,
    norm,
)

# Reuse the *tuned* solver kernels and per-regime configuration from the
# single-shot experiment script so both stay on one configuration.  Only
# the mesh construction (which must vary with the refinement level) and
# the error/rate post-processing are re-implemented here.
from helicity_experiments import (
    bc_config,
    bcs_for_A,
    bcs_for_B,
    build_function_spaces,
    initial_field,
    order,
    solve_constrained_div_free,
    solve_curl_curl,
)
from mesh_independence import beta_curl, gamma, spp_helicity, spp_riesz

dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

# ----------------------------------------------------------------------
# Analytic helicity of each regime's analytic initial field.
# ----------------------------------------------------------------------
ANALYTIC_HELICITY = {
    "closed":    -32.0 / (3.0 * math.pi ** 2),   # ∫ A₀·curl A₀  ≈ −1.0808
    "line-tied": 1.0,                            # H_R = α/108, α = 108
    "periodic":  -8.0 / (math.pi ** 2),          # ∫ A·(B+B_H) = −2∫ψ ≈ −0.8106
}


def is_root():
    return COMM_WORLD.rank == 0


def rprint(*a, **k):
    if is_root():
        print(*a, **k)


# ----------------------------------------------------------------------
# Mesh construction parametrised by refinement level.
#
# `level` MG refinements on a UnitSquareMesh(n_base) base, extruded with
# `n_layers_base` layers on the coarsest mesh.  ExtrudedMeshHierarchy
# refines the vertical direction in lock-step, so at level k the finest
# mesh is  (n_base·2^k) × (n_base·2^k) × (n_layers_base·2^k)  prisms and
# the mesh size is  h = 1 / (n_base · 2^k).
# ----------------------------------------------------------------------
def build_mesh_at_level(periodic, level, n_base, n_layers_base):
    base = UnitSquareMesh(n_base, n_base, distribution_parameters=dparams)
    base_mh = MeshHierarchy(base, level, distribution_parameters=dparams)
    builder = partial(ExtrudedMesh, periodic=True) if periodic else ExtrudedMesh
    mh = ExtrudedMeshHierarchy(base_mh, height=1.0,
                               base_layer=n_layers_base, mesh_builder=builder)
    return mh[-1]


# ----------------------------------------------------------------------
# One solve at one (regime, level): returns the computed helicity plus
# all iteration counts / DOFs / curl residuals.  This mirrors the solve
# pipeline of helicity_experiments.run_experiment, minus the ParaView I/O.
# ----------------------------------------------------------------------
def compute_helicity(bc_mode, mesh, a_variant="lateral"):
    periodic, needs_reference, _ = bc_config(bc_mode)
    Vc, Vd, Vn = build_function_spaces(mesh)

    X, Y, Z = SpatialCoordinate(mesh)
    B_init = initial_field(bc_mode, X, Y, Z)

    label = bc_mode if bc_mode != "line-tied" else f"line-tied_{a_variant}"

    # ---- Reference field B' (line-tied only; vacuum reference = 0 else) ----
    if needs_reference:
        Z_ref = MixedFunctionSpace([Vd, Vn])
        bcs_ref = bcs_for_B(Z_ref.sub(0), B_init, bc_mode)
        B_prime, its_Bp = solve_constrained_div_free(
            Vd, Vn, bcs_ref,
            options_prefix=f"{label}_Bprime", has_nullspace=True)
        dofs_ref = Z_ref.dim()
    else:
        B_prime = Function(Vd, name="B_reference")
        dofs_ref = 0
        its_Bp = 0

    # ---- Divergence-free projection of B_init ----
    Z_proj = MixedFunctionSpace([Vd, Vn])
    bcs_proj = bcs_for_B(Z_proj.sub(0), B_init, bc_mode)
    B_func, its_B = solve_constrained_div_free(
        Vd, Vn, bcs_proj,
        options_prefix=f"{label}_Bproj", source=B_init, has_nullspace=True)
    dofs_proj = Z_proj.dim()

    # ---- Companion vector potential A' (line-tied only) ----
    if bc_mode == "line-tied":
        A_prime, its_Ap, _ = solve_curl_curl(
            Vc, B_prime, bcs=[],
            options_prefix=f"{label}_Aprime", name="A_prime")
        err_curlAp = float(norm(curl(A_prime) - B_prime, "L2"))
    else:
        A_prime = None
        its_Ap = 0
        err_curlAp = 0.0

    # ---- H(curl) helicity solve: curl A = B_func - B_prime ----
    B_field = B_func - B_prime
    A, its_A, _ = solve_curl_curl(
        Vc, B_field, bcs=bcs_for_A(Vc, bc_mode, variant=a_variant),
        options_prefix=f"{label}_helicity", name="A")
    err_curlA = float(norm(curl(A) - B_field, "L2"))

    # ---- Regime-specific helicity integral (Table 2.1 of the thesis) ----
    if bc_mode == "periodic":
        B_H = B_func - curl(A)
        H = assemble(inner(A, B_func + B_H) * dx)
    elif bc_mode == "line-tied":
        H = assemble(inner(A + 2.0 * A_prime, B_func - B_prime) * dx)
    else:  # closed
        H = assemble(inner(A, B_func) * dx)

    return {
        "H":          float(H),
        "its_Bp":     its_Bp,
        "its_B":      its_B,
        "its_Ap":     its_Ap,
        "its_A":      its_A,
        "err_curlA":  err_curlA,
        "err_curlAp": err_curlAp,
        "dofs_A":     Vc.dim(),
        "dofs_ref":   dofs_ref,
        "dofs_proj":  dofs_proj,
    }


# ----------------------------------------------------------------------
# Convergence sweep for one regime over the requested refinement levels.
# ----------------------------------------------------------------------
def run_convergence(bc_mode, levels, n_base, n_layers_base, a_variant):
    periodic, _, _ = bc_config(bc_mode)
    H_exact = ANALYTIC_HELICITY[bc_mode]

    rprint(f"\n{'#' * 70}")
    rprint(f"#  Regime: {bc_mode}   "
           f"(periodic_z={periodic}, H_analytic={H_exact:.10e})")
    rprint(f"{'#' * 70}", flush=True)

    # h halves every level → consecutive-level rate = log2(e_{k-1}/e_k).
    def _rate(prev, cur):
        if prev is None or prev == 0.0 or cur == 0.0:
            return float("nan")
        return math.log(prev / cur) / math.log(2.0)

    rows = []
    prev_err = None
    prev_err_curlA = None
    for k in levels:
        N_fine = n_base * 2 ** k
        h = 1.0 / N_fine
        mesh = build_mesh_at_level(periodic, k, n_base, n_layers_base)
        res = compute_helicity(bc_mode, mesh, a_variant=a_variant)

        abs_err = abs(res["H"] - H_exact)
        rel_err = abs_err / abs(H_exact) if H_exact != 0.0 else float("nan")
        rate = _rate(prev_err, abs_err)
        prev_err = abs_err
        # Curl-reconstruction error ||B - curl A||_L2 and its rate.  For
        # closed / line-tied this decays at O(h); for periodic it plateaus
        # near 1 because B - curl A ≈ ê_z is the harmonic 1-form residual
        # (the topological mode that is closed but not exact, hence never
        # in the image of curl), so its rate ≈ 0 by construction.
        rate_curlA = _rate(prev_err_curlA, res["err_curlA"])
        prev_err_curlA = res["err_curlA"]

        row = {
            "regime":     bc_mode,
            "level":      k,
            "N":          N_fine,
            "h":          h,
            "dofs_A":     res["dofs_A"],
            "dofs_ref":   res["dofs_ref"],
            "dofs_proj":  res["dofs_proj"],
            "its_Bp":     res["its_Bp"],
            "its_B":      res["its_B"],
            "its_Ap":     res["its_Ap"],
            "its_A":      res["its_A"],
            "H_computed": res["H"],
            "H_analytic": H_exact,
            "abs_error":  abs_err,
            "rel_error":  rel_err,
            "rate":       rate,
            "err_curlA":  res["err_curlA"],
            "rate_curlA": rate_curlA,
            "err_curlAp": res["err_curlAp"],
        }
        rows.append(row)

        rate_str = "  —  " if math.isnan(rate) else f"{rate:5.2f}"
        rprint(f"[done] {bc_mode:<10} level={k} N={N_fine:<4} "
               f"h={h:.4f}  dofs(A)={res['dofs_A']:<9} "
               f"iters(B'/B/A'/A)={res['its_Bp']}/{res['its_B']}/"
               f"{res['its_Ap']}/{res['its_A']}  "
               f"H={res['H']:+.8e}  |err|={abs_err:.4e}  rate={rate_str}  "
               f"||B-curlA||={res['err_curlA']:.4e}",
               flush=True)

    return rows


# ----------------------------------------------------------------------
# Output: terminal convergence table, CSV, and param.txt.
# ----------------------------------------------------------------------
def print_convergence_table(bc_mode, rows):
    H_exact = ANALYTIC_HELICITY[bc_mode]
    cols = ("level", "N", "h", "dofs(A)", "iters(A)",
            "H_computed", "abs_err", "rel_err", "rate",
            "||B-curlA||", "rate_c")
    widths = (5, 5, 9, 10, 9, 18, 12, 12, 6, 12, 6)
    header = "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    total = sum(widths) + 2 * (len(widths) - 1)

    rprint()
    rprint("=" * total)
    rprint(f"Convergence — {bc_mode}   (H_analytic = {H_exact:+.10e})")
    rprint("=" * total)
    rprint(header)
    rprint("-" * total)
    for r in rows:
        rate = "   —  " if math.isnan(r["rate"]) else f"{r['rate']:6.2f}"
        rate_c = "   —  " if math.isnan(r["rate_curlA"]) else f"{r['rate_curlA']:6.2f}"
        rprint(f"{r['level']:>{widths[0]}}  "
               f"{r['N']:>{widths[1]}}  "
               f"{r['h']:>{widths[2]}.4f}  "
               f"{r['dofs_A']:>{widths[3]}}  "
               f"{r['its_A']:>{widths[4]}}  "
               f"{r['H_computed']:>{widths[5]}.10e}  "
               f"{r['abs_error']:>{widths[6]}.4e}  "
               f"{r['rel_error']:>{widths[7]}.4e}  "
               f"{rate:>{widths[8]}}  "
               f"{r['err_curlA']:>{widths[9]}.4e}  "
               f"{rate_c:>{widths[10]}}")


CSV_FIELDS = ("regime", "level", "N", "h",
              "dofs_A", "dofs_ref", "dofs_proj",
              "its_Bp", "its_B", "its_Ap", "its_A",
              "H_computed", "H_analytic", "abs_error", "rel_error", "rate",
              "err_curlA", "rate_curlA", "err_curlAp")


def write_csv(path, all_rows):
    if not is_root():
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in CSV_FIELDS})


def write_params(path, params, header=""):
    """Print all reproducibility parameters to the terminal AND param.txt."""
    if not is_root():
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git = "unknown"
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [f"# {header}", f"# time: {ts}", f"# git:  {git}", ""]
    width = max(len(k) for k in params) + 2
    for k, v in params.items():
        lines.append(f"{k:<{width}} {v}")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(path, "w") as f:
        f.write(text)


def main():
    p = argparse.ArgumentParser(
        description="Helicity convergence study (analytic vs computed).")
    p.add_argument("--regimes", nargs="+", default=["closed", "line-tied", "periodic"],
                   choices=["closed", "line-tied", "periodic"],
                   help="Which BC regimes to run (default: all three).")
    p.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4],
                   help="Mesh refinement levels (default: 1 2 3 4).")
    p.add_argument("--n-base", type=int, default=4,
                   help="Horizontal cells per side on the coarsest base mesh.")
    p.add_argument("--n-layers", type=int, default=4,
                   help="Vertical layers on the coarsest extruded mesh.")
    p.add_argument("--a-variant", default="lateral",
                   choices=["lateral", "full", "topbot", "none"],
                   help="A×n BC variant for the line-tied regime.")
    p.add_argument("--outdir", default="output_convergence",
                   help="Output directory for CSV / param.txt.")
    args = p.parse_args()

    levels = sorted(set(args.levels))

    # --- Reproducibility record: print to terminal AND write param.txt. ---
    finest = {bc: f"{args.n_base * 2**max(levels)}^2 x "
                  f"{args.n_layers * 2**max(levels)}"
              for bc in args.regimes}
    params = {
        "script":            os.path.basename(__file__),
        "order":             order,
        "regimes":           ",".join(args.regimes),
        "levels":            ",".join(map(str, levels)),
        "n_base":            args.n_base,
        "n_layers":          args.n_layers,
        "finest_NxNxNz":     "; ".join(f"{bc}:{v}" for bc, v in finest.items()),
        "a_variant":         args.a_variant,
        "element_Vc":        f"N1curl x CG / CG x DG (prism, order {order})",
        "element_Vd":        f"RT x DG / DG x CG (prism, order {order})",
        "element_Vn":        f"DG x DG (prism, order {order})",
        "beta_curl":         float(beta_curl),
        "gamma":             float(gamma),
        "H_closed":          ANALYTIC_HELICITY["closed"],
        "H_line-tied":       ANALYTIC_HELICITY["line-tied"],
        "H_periodic":        ANALYTIC_HELICITY["periodic"],
        "helicity_ksp":      spp_helicity["ksp_type"],
        "helicity_pc":       f"{spp_helicity['pc_type']}({spp_helicity['pc_python_type']})",
        "helicity_ksp_max_it": spp_helicity["ksp_max_it"],
        "riesz_ksp":         spp_riesz["ksp_type"],
        "riesz_ksp_max_it":  spp_riesz["ksp_max_it"],
        "riesz_ksp_rtol":    spp_riesz["ksp_rtol"],
        "mpi_ranks":         COMM_WORLD.size,
        "outdir":            args.outdir,
    }
    write_params(os.path.join(args.outdir, "param.txt"),
                 params, header=os.path.basename(__file__))

    all_rows = []
    per_regime = {}
    for bc in args.regimes:
        rows = run_convergence(bc, levels, args.n_base, args.n_layers,
                               args.a_variant)
        per_regime[bc] = rows
        all_rows.extend(rows)

    if is_root():
        for bc in args.regimes:
            print_convergence_table(bc, per_regime[bc])
        csv_path = os.path.join(args.outdir, "helicity_convergence.csv")
        write_csv(csv_path, all_rows)
        print(f"\nWrote {csv_path}")
        print(f"Wrote {os.path.join(args.outdir, 'param.txt')}")


if __name__ == "__main__":
    main()
