# AGENTS.md

Guía operativa para agentes de IA y colaboradores que trabajen en este
repositorio. El proyecto es académico, está en español y busca ser
reproducible y fácil de extender.

## Resumen

Aplicación de escritorio en Python con cinco metaheurísticas implementadas
desde cero (GA, ACO, PSO, AIS/CLONALG y DE), problemas continuos, binarios,
TSP y un caso real de reparto en Zapopan. La interfaz usa CustomTkinter y
Matplotlib; el motor de optimización corre en un hilo secundario.

## Entorno y comandos

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python main.py
python run_experiments.py --quick
python run_experiments.py
python -m compileall -q main.py core problems algorithms ui
```

- Python probado: **3.13.3** en Windows. Tkinter debe estar disponible.
- Dependencias directas: `customtkinter`, `matplotlib`, `numpy`.
- **Prohibido** usar `scipy.optimize`, `deap`, `pyswarms` o librerías de
  metaheurísticas: los algoritmos son propios.
- El backend de Matplotlib en scripts debe ser `Agg`; en la UI es `TkAgg`.

## Mapa de responsabilidades

```
core/base.py       Contratos, ParamSpec, Snapshot, operadores, plantilla run()
core/engine.py     Hilos, cola de eventos, pausa/parada/reinicio, run_sync
problems/*.py      Definición, evaluación, validación y visualización de datos
algorithms/*.py    Las cinco metaheurísticas, sin imports de Tkinter
ui/app.py          Toda la capa CustomTkinter y los dibujos Matplotlib
data/*.json        Dataset de 12 nodos de Zapopan
run_experiments.py Figuras y resultados reproducibles
```

Regla de dependencias: `problems`, `algorithms` y `core` **no** deben importar
`ui` ni `tkinter`. La UI consume los registros y los esquemas; no debe
contener lógica matemática duplicada.

## Contratos que no se rompen

### `Problem` (`core/base.py`)

- Atributos de clase: `key`, `name`, `family` (`continuous`, `binary`,
  `permutation`), `sense` (`min`/`max`).
- Métodos: `param_schema`, `evaluate`, `random_solution`, `validate`,
  `diversity`, `describe` y, para permutaciones, los ganchos de ACO
  (`aco_distance_matrix`, `aco_initial_weights`, `aco_edges`).
- `evaluate` debe devolver un `float` finito en la escala del problema.
- `validate` devuelve `(bool, mensaje)` y se usa en pruebas y UI.
- `describe` alimenta el panel de información; no debe lanzar excepciones.

### `MetaheuristicAlgorithm`

- Implementar `initialize()` y `step()`; **no** reescribir `run()` salvo causa
  justificada.
- `initialize()` fija `self.population` y puede fijar `self.fitness`. Si no lo
  hace, la clase base evalúa la población una sola vez.
- Cada `step()` debe actualizar `self.population` y `self.fitness` evaluando
  solo los individuos nuevos (no re-evaluar toda la población).
- Usar `self.evaluate_individual`, `self.problem.is_better` y
  `self.refresh_best`; no comparar fitness con `<`/`>` directamente.
- Declarar `family_support`; ACO solo admite `permutation`.
- Los parámetros se declaran con `ParamSpec` y se resuelven con
  `resolve_default(family)` / `resolve_choices(family)`. La UI depende de esto.

### Snapshots y hilo

- La UI nunca toca el algoritmo desde el hilo de Tkinter. El motor publica
  `EngineEvent` en una cola; la UI usa `root.after` para drenarla.
- Detener y pausar son cooperativos: el bucle comprueba los eventos en cada
  iteración.
- Los snapshots guardan mejor solución, media, diversidad, evaluaciones y
  tiempo; el historial debe servir para reproducir la solución en la UI.

## Convenciones de código

- Identificadores en inglés; cadenas de UI, docstrings y comentarios en
  español.
- Type hints en funciones y métodos públicos.
- Usar el `np.random.Generator` de la instancia (`self.rng`), nunca el RNG
  global de NumPy. La semilla entra por el constructor.
- Nada de estado global mutable entre corridas.
- Las soluciones se reparan antes de evaluarse: clip para continuo, bits para
  binario, `argsort`/permutación para Random-Key.
- Comentarios didácticos en los algoritmos: explicar selección, cruza,
  mutación, feromonas, sigmoide o Random-Key.
- Mantener el estilo de Matplotlib y los colores usados en `run_experiments.py`
  para que las figuras sean consistentes.

## Verificación antes de cerrar un cambio

1. `python -m compileall -q main.py core problems algorithms ui`.
2. Matriz headless de todas las combinaciones válidas (pocas iteraciones) y
   validar cada mejor solución con `problem.validate`.
3. Comprobar reproducibilidad: misma semilla ⇒ mismo mejor objetivo.
4. Probar la GUI: iniciar, pausar, reanudar, detener, reiniciar y cambiar de
   problema/algoritmo.
5. Ejecutar `python run_experiments.py --quick` y confirmar que se generan
   `fig-3.png` … `fig-10.png`.
6. Revisar que no se haya añadido una dependencia no permitida ni un import de
   Tkinter en `core/`, `problems/` o `algorithms/`.

## Git

- Rama principal: `main`.
- Commits atómicos con Conventional Commits en español, por ejemplo:
  `feat: motor de ejecución en hilo secundario`.
- No subir `.venv/`, `__pycache__/`, logs ni credenciales.
- Las figuras `fig-3.png` … `fig-10.png` y los CSV/JSON de resultados sí se
  versionan como entregables.
- Tag de entrega: `v1.0.0`.

## Cómo extender

### Nuevo problema

1. Crear una clase en `problems/` que herede de `ContinuousProblem`,
   `BinaryProblem` o `PermutationProblem`.
2. Definir `key`, `name`, `family`, `sense`, `param_schema`, `evaluate`,
   `validate` (si aplica) y `describe`.
3. Registrar la clase en `problems/__init__.py` (`PROBLEMS` y `PROBLEM_ORDER`).
4. Añadir la visualización en `ui/app.py` si la familia no está cubierta.
5. Añadir un escenario en `run_experiments.py` solo si aporta una figura.

### Nuevo algoritmo

1. Crear la clase en `algorithms/` heredando de `MetaheuristicAlgorithm`.
2. Declarar `key`, `name`, `family_support`, `description` y `param_schema`.
3. Implementar `initialize()` y `step()` usando los operadores de
   `core/base.py`.
4. Registrarlo en `algorithms/__init__.py` y en `ALGORITHM_ORDER`.
5. Verificar todas las familias declaradas y añadir la etiqueta/color en
   `run_experiments.py` si se compara en figuras.

## Trampas conocidas

- **Tkinter no es thread-safe**: ningún callback del motor debe actualizar
  widgets. Usar la cola y `after`.
- **Matplotlib**: no llamar `plt.show()` en scripts; usar `Agg`. En la UI,
  dibujar solo en el hilo principal y con `draw_idle`.
- **CTkSlider**: `from_` y `to` no pueden ser iguales; usar `to=1` cuando no
  hay historial.
- **ACO**: no tiene sentido en continuo/binario; la UI lo desactiva y el
  motor lanza `ValueError`.
- **Reparto**: el decodificador divide por capacidad; una demanda individual
  mayor que `Q_max` debe rechazarse en `validate`.
- **Finalización**: al cerrar la ventana hay que detener el motor antes de
  destruir Tk.
- **CRLF**: el repo usa `.gitattributes` con LF; no reintroducir CRLF.
- **Resultados**: no editar las figuras a mano; regenerarlas con el script.

## Entrega

- `README.md` explica instalación, uso, problemas, algoritmos y figuras.
- No hay archivo `LICENSE`; no afirmar que el proyecto es de código abierto.
- Repositorio remoto: `entorno-metaheuristicas` (público).
