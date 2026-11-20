"""Prueba robusta del control pausar/reanudar/detener en la GUI."""

from __future__ import annotations

import time

from ui.app import MetaheuristicApp


def pump(app: MetaheuristicApp, seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        app.update()
        time.sleep(0.01)


def main() -> None:
    app = MetaheuristicApp()
    app.iterations_var.set("100000")
    app.snapshot_var.set("10")
    app._start_run(False)

    pump(app, 0.6)
    print("tras iniciar: snapshots =", len(app.snapshots), flush=True)
    assert len(app.snapshots) >= 1

    app._on_pause()
    pump(app, 0.15)
    n1 = len(app.snapshots)
    pump(app, 0.8)
    n2 = len(app.snapshots)
    print(f"pausa: n1={n1} n2={n2} is_paused={app.engine.is_paused}", flush=True)
    assert n2 - n1 <= 1, f"la pausa no detuvo la GUI: {n1} -> {n2}"

    app._on_pause()
    pump(app, 0.5)
    n3 = len(app.snapshots)
    print(f"reanudar: n3={n3} is_paused={app.engine.is_paused}", flush=True)
    assert n3 > n2, "no se reanudó"

    app._on_stop()
    pump(app, 0.8)
    print(
        "detener: estado=",
        app.status_label.cget("text"),
        "ui_running=",
        app._ui_running,
        "is_running=",
        app.engine.is_running,
        flush=True,
    )
    assert not app.engine.is_running
    assert not app._ui_running

    app._on_close()
    print("GUI CONTROL OK", flush=True)


if __name__ == "__main__":
    main()