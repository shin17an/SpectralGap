"""Reproduce the short-time Galerkin illustration used in the manuscript.

The initial datum is independent of x_2, so the two-dimensional Fourier
Galerkin system leaves this one-dimensional subspace invariant.  The code
uses a symmetric Störmer--Verlet method and exact convolution in Fourier
space.  It has no third-party dependencies.
"""

from __future__ import annotations

import csv
import math
import struct
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
MASS = 1.0
LAMBDA = 1.0
AMPLITUDE = 0.1
INITIAL_BAND = 1
MAX_MODE = 15
DT = 5.0e-4
FINAL_TIME = 1.2
SAMPLE_COUNT = 121
C_EMBED = 1.0 + 6.0 * math.sqrt(2.0)
EPSILON = AMPLITUDE  # omega_1^2=2 and ||A cos x||_2^2=A^2/2.


def modes(max_mode: int) -> range:
    return range(-max_mode, max_mode + 1)


def cubic_coefficients(u: dict[int, complex], max_mode: int) -> dict[int, complex]:
    square: dict[int, complex] = {}
    for p in modes(max_mode):
        up = u[p]
        for q in modes(max_mode):
            square[p + q] = square.get(p + q, 0.0j) + up * u[q]
    cube: dict[int, complex] = {}
    for k in modes(max_mode):
        value = 0.0j
        for s, usq in square.items():
            r = k - s
            if -max_mode <= r <= max_mode:
                value += usq * u[r]
        cube[k] = value
    return cube


def acceleration(u: dict[int, complex], max_mode: int) -> dict[int, complex]:
    cube = cubic_coefficients(u, max_mode)
    return {
        k: -(k * k + MASS * MASS) * u[k] - LAMBDA * cube[k]
        for k in modes(max_mode)
    }


def quadratic_tail(
    u: dict[int, complex], v: dict[int, complex], cutoff: int
) -> float:
    return 0.5 * sum(
        abs(v[k]) ** 2 + (k * k + MASS * MASS) * abs(u[k]) ** 2
        for k in u
        if abs(k) > cutoff
    )


def hamiltonian(u: dict[int, complex], v: dict[int, complex], max_mode: int) -> float:
    quadratic = 0.5 * sum(
        abs(v[k]) ** 2 + (k * k + MASS * MASS) * abs(u[k]) ** 2
        for k in modes(max_mode)
    )
    cube = cubic_coefficients(u, max_mode)
    integral_u4 = sum(cube[k] * u[-k] for k in modes(max_mode)).real
    return quadratic + 0.25 * LAMBDA * integral_u4


def theorem_bound(t: float, interaction_index: int) -> float:
    q = 2.0 * abs(LAMBDA) * C_EMBED * EPSILON**2 * abs(t)
    if q >= 1.0:
        return math.inf
    return EPSILON**2 * q ** (2 * interaction_index + 2) / (2.0 * (1.0 - q))


def simulate(dt: float, sample_count: int = SAMPLE_COUNT):
    u = {k: 0.0j for k in modes(MAX_MODE)}
    v = {k: 0.0j for k in modes(MAX_MODE)}
    u[-INITIAL_BAND] = AMPLITUDE / 2.0
    u[INITIAL_BAND] = AMPLITUDE / 2.0
    initial_h = hamiltonian(u, v, MAX_MODE)

    sample_steps = {
        round(j * FINAL_TIME / (sample_count - 1) / dt): j
        for j in range(sample_count)
    }
    total_steps = round(FINAL_TIME / dt)
    rows = []

    for step in range(total_steps + 1):
        if step in sample_steps:
            t = step * dt
            actual = [quadratic_tail(u, v, 2 * n + 1) for n in range(3)]
            bounds = [theorem_bound(t, n) for n in range(3)]
            rows.append((t, *actual, *bounds))
        if step == total_steps:
            break
        a0 = acceleration(u, MAX_MODE)
        v_half = {k: v[k] + 0.5 * dt * a0[k] for k in modes(MAX_MODE)}
        u = {k: u[k] + dt * v_half[k] for k in modes(MAX_MODE)}
        a1 = acceleration(u, MAX_MODE)
        v = {k: v_half[k] + 0.5 * dt * a1[k] for k in modes(MAX_MODE)}

    final_h = hamiltonian(u, v, MAX_MODE)
    relative_drift = abs(final_h - initial_h) / abs(initial_h)
    return rows, relative_drift


def write_csv(rows) -> None:
    with (HERE / "transfer_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t",
                "actual_N0",
                "actual_N1",
                "actual_N2",
                "bound_N0",
                "bound_N1",
                "bound_N2",
            ]
        )
        writer.writerows(rows)


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_plot(rows) -> None:
    width, height = 540.0, 330.0
    left, right, bottom, top = 62.0, 20.0, 48.0, 24.0
    plot_w, plot_h = width - left - right, height - bottom - top
    y_min, y_max = -30.0, -2.0

    def xp(t: float) -> float:
        return left + plot_w * t / FINAL_TIME

    def yp(value: float) -> float:
        log_value = math.log10(max(value, 1.0e-30))
        return bottom + plot_h * (log_value - y_min) / (y_max - y_min)

    content: list[str] = []

    def text(x: float, y: float, size: float, value: str) -> None:
        content.append(
            f"BT /F1 {size:.1f} Tf {x:.2f} {y:.2f} Td ({pdf_escape(value)}) Tj ET"
        )

    content.append("1 1 1 rg 0 0 540 330 re f")
    content.append("0 0 0 rg")
    content.append("0.85 0.85 0.85 RG 0.4 w")
    for exponent in range(-30, -1, 5):
        y = bottom + plot_h * (exponent - y_min) / (y_max - y_min)
        content.append(f"{left:.2f} {y:.2f} m {left + plot_w:.2f} {y:.2f} l S")
        text(19.0, y - 3.0, 8.0, f"1e{exponent}")
    for j in range(7):
        t = FINAL_TIME * j / 6.0
        x = xp(t)
        content.append(f"{x:.2f} {bottom:.2f} m {x:.2f} {bottom + plot_h:.2f} l S")
        text(x - 7.0, 31.0, 8.0, f"{t:.1f}")

    content.append("0 0 0 RG 0.8 w")
    content.append(
        f"{left:.2f} {bottom:.2f} m {left + plot_w:.2f} {bottom:.2f} l "
        f"{left + plot_w:.2f} {bottom + plot_h:.2f} l {left:.2f} {bottom + plot_h:.2f} l h S"
    )
    text(width / 2.0 - 18.0, 13.0, 9.0, "time t")
    text(9.0, height - 16.0, 8.0, "quadratic tail energy")

    colors = [(0.10, 0.35, 0.75), (0.80, 0.20, 0.15), (0.10, 0.55, 0.25)]
    for n, (red, green, blue) in enumerate(colors):
        for column, dashed in ((1 + n, False), (4 + n, True)):
            content.append(f"{red:.2f} {green:.2f} {blue:.2f} RG 1.15 w")
            content.append("[5 3] 0 d" if dashed else "[] 0 d")
            points = [(xp(row[0]), yp(row[column])) for row in rows]
            x0, y0 = points[0]
            content.append(f"{x0:.2f} {y0:.2f} m")
            for x, y in points[1:]:
                content.append(f"{x:.2f} {y:.2f} l")
            content.append("S")
    content.append("[] 0 d")

    legend_y = height - 17.0
    for n, (red, green, blue) in enumerate(colors):
        x = 175.0 + 108.0 * n
        content.append(f"{red:.2f} {green:.2f} {blue:.2f} RG 1.15 w")
        content.append(f"{x:.2f} {legend_y:.2f} m {x + 14:.2f} {legend_y:.2f} l S")
        content.append("[5 3] 0 d")
        content.append(f"{x:.2f} {legend_y - 7:.2f} m {x + 14:.2f} {legend_y - 7:.2f} l S")
        content.append("[] 0 d")
        text(x + 18.0, legend_y - 5.0, 7.5, f"N={n}: actual / bound")

    stream = "\n".join(content).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 540 330] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    (HERE / "transfer_comparison.pdf").write_bytes(output)


def write_summary(rows, drift: float, convergence_error: float) -> None:
    final = rows[-1]

    def tex_scientific(value: float, digits: int) -> str:
        mantissa, exponent = f"{value:.{digits}e}".split("e")
        return f"{mantissa}\\mathbin{{\\times}}10^{{{int(exponent)}}}"

    with (HERE / "numerical_summary.tex").open("w", encoding="ascii") as handle:
        handle.write("% Generated by simulate_transfer.py.\n")
        handle.write(
            f"\\newcommand{{\\NumericalEnergyDrift}}{{{tex_scientific(drift, 2)}}}\n"
        )
        handle.write(
            f"\\newcommand{{\\NumericalConvergence}}{{{tex_scientific(convergence_error, 2)}}}\n"
        )
        handle.write("\\newcommand{\\NumericalTable}{%\n")
        handle.write("\\begin{tabular}{@{}cccc@{}}\n")
        handle.write("\\toprule\n")
        handle.write("$N$ & cutoff & computed tail & theorem bound \\\\\n")
        handle.write("\\midrule\n")
        for n in range(3):
            handle.write(
                f"{n} & ${(2*n+1)}K$ & ${tex_scientific(final[1+n], 3)}$ "
                f"& ${tex_scientific(final[4+n], 3)}$ \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}%\n")
        handle.write("}\n")


def main() -> None:
    rows, drift = simulate(DT)
    coarse_rows, _ = simulate(2.0 * DT)
    convergence_error = max(
        abs(rows[-1][j] - coarse_rows[-1][j]) for j in range(1, 4)
    )
    write_csv(rows)
    write_plot(rows)
    write_summary(rows, drift, convergence_error)
    print(f"samples={len(rows)}")
    print(f"relative Hamiltonian drift={drift:.3e}")
    print(f"fine/coarse final-tail difference={convergence_error:.3e}")
    print("final row:", rows[-1])


if __name__ == "__main__":
    main()
