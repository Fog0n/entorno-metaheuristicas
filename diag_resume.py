"""Diagnóstico: ¿qué le pasa al hilo del motor al reanudar?"""

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

    pump(app, 0.5)
    print("inicio: snaps", len(app.snapshots), "running", app.engine.is_running,
          "thread", app.engine._thread.is_alive() if app.engine._thread else None, flush=True)

    app._on_pause()
    pump(app, 0.3)
    print("pausa: snaps", len(app.snapshots), "paused", app.engine.is_paused,
          "thread alive", app.engine._thread.is_alive() if app.engine._thread else None, flush=True)

    app._on_pause()  # reanudar
    pump(app, 0.2)
    print("reanudo inmediato: paused", app.engine.is_paused,
          "thread alive", app.engine._thread.is_alive() if app.engine._thread else None, flush=True)
    pump(app, 0.8)
    print("reanudo +0.8s: snaps", len(app.snapshots),
          "thread alive", app.engine._thread.is_alive() if app.engine._thread else None,
          "status:", app.status_label.cget("text"), flush=True)

    # Eventos crudos pendientes en la cola del motor
    pending = [e.kind for e in app.engine.drain()]
    print("eventos crudos pendientes:", pending[:10], "total", len(pending), flush=True)

    app._on_close()


if __name__ == "__main__":
    main()