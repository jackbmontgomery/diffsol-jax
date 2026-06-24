import re
from pathlib import Path

import bench_forward
import bench_grad
import bench_stiff

DOCS = Path(__file__).resolve().parent.parent / "docs" / "benchmarks.md"


def fmt_ms(ms: float) -> str:
    return f"{ms:.2f} ms" if ms < 10 else f"{ms:.1f} ms"


def forward_table(rows: list[dict]) -> str:
    header = [
        "| System                     | diffsol-jax | diffrax            | Speedup |",
        "| -------------------------- | ----------- | ------------------ | ------- |",
    ]
    body = [
        f"| {r['label']} | {fmt_ms(r['ds_ms'])} | {fmt_ms(r['dx_ms'])} ({r['dx_solver']}) | {r['speedup']:.2f}x |"
        for r in rows
    ]
    return "\n".join(header + body)


def grad_table(r: dict, n_steps: int) -> str:
    return "\n".join(
        [
            "| System                       | diffsol-jax           | diffrax               | Speedup |",
            "| ---------------------------- | --------------------- | --------------------- | ------- |",
            f"| {r['label']} | {fmt_ms(r['ds_ms'])} / {n_steps} steps | {fmt_ms(r['dx_ms'])} / {n_steps} steps | {r['speedup']:.2f}x |",
        ]
    )


def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(
        rf"(<!-- BENCH:{name}:start -->).*?(<!-- BENCH:{name}:end -->)",
        re.DOTALL,
    )
    if not pat.search(text):
        raise SystemExit(f"marker BENCH:{name} not found in {DOCS}")
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text)


def main() -> None:
    print("running forward benchmark...")
    fwd = bench_forward.run()
    print("running stiff benchmark...")
    stf = bench_stiff.run()
    print("running gradient benchmark...")
    grd = bench_grad.run()

    text = DOCS.read_text()
    text = replace_block(text, "forward", forward_table([fwd, stf]))
    text = replace_block(text, "grad", grad_table(grd, bench_grad.N_STEPS))
    DOCS.write_text(text)

    print(f"\nupdated {DOCS.relative_to(DOCS.parents[1])}")
    print(f"  forward LV : {fwd['speedup']:.2f}x  (max |diff| {fwd['max_diff']:.1e})")
    print(f"  stiff  VdP : {stf['speedup']:.2f}x  (max |diff| {stf['max_diff']:.1e})")
    print(f"  grad   LV  : {grd['speedup']:.2f}x  (p_err {grd['ds_err']:.4f})")


if __name__ == "__main__":
    main()
