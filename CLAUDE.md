# CLAUDE.md

## Project Overview

This is a **Particle Swarm Optimization (PSO)** library written in Go. It provides a general-purpose optimizer that uses swarm intelligence to minimize objective functions, supporting bounds, constraints, and parallel fitness evaluation.

- **Module**: `github.com/quells/pso`
- **Go version**: 1.24+ (depends on `golang.org/x/sync`)
- **Default branch**: `main`

## Repository Structure

```
pso/
├── go.mod                        # Go module definition
├── pkg/
│   └── swarm/
│       └── swarm.go              # Core PSO library (Optimizer, Range, Options, fitness types)
└── example/
    ├── golinski/
    │   └── golinski.go           # Golinski speed reducer benchmark optimization
    └── iris/
        ├── iris.go               # Iris classification using PSO-trained neural network
        ├── nn.go                 # Feed-forward neural network (4-4-4-3 architecture)
        ├── data.go               # Iris dataset loader (embedded CSV)
        └── iris.txt              # Raw Iris dataset
```

## Build and Run Commands

There is no Makefile or CI configuration. Use standard Go tooling:

```bash
# Build the library
go build ./pkg/...

# Vet the code
go vet ./...

# Run an example
go run ./example/golinski/
go run ./example/iris/
```

There are currently **no tests** in this repository (`*_test.go` files do not exist).

## Core Library (`pkg/swarm`)

### Key Types

- **`Fitness`** — `func([]float64) float64` — Objective function to minimize. Return negative values to maximize.
- **`Constraint`** — `func([]float64) bool` — Returns false if a particle position is invalid.
- **`Range`** — `[2]float64` — Defines `[lower, upper]` bounds for a dimension. Zero value means unbounded.
- **`Options`** — Configuration struct with hyperparameters, bounds, constraints, parallelism settings.
- **`Optimizer`** — The main PSO engine. Created via `New()`, driven by `Step()` or `StepUntil()`.

### API Surface

| Function/Method | Purpose |
|---|---|
| `New(fitness, shape, options)` | Create a new optimizer. Shape initializes particle positions (does not enforce bounds). |
| `opt.Reset()` | Re-randomize all particle positions and velocities. |
| `opt.Step(ctx)` | Execute one optimization step. Returns error on context cancellation. |
| `opt.StepUntil(ctx, progressRate)` | Run steps until convergence or context cancellation. Returns `(steps, error)`. |
| `opt.Best()` | Return the best position found so far. |

### Default Hyperparameters

| Parameter | Default | Purpose |
|---|---|---|
| `LocalSize` | 25 | Size of particle neighborhoods |
| `PopulationSize` | `10 * LocalSize * dimensions` | Total particle count |
| `Parallelism` | `GOMAXPROCS` | Worker goroutine count |
| `Inertia` | 0.95 | Velocity dampening |
| `ParticleStep` | 0.75 | Pull toward particle's own best |
| `LocalStep` | 0.50 | Pull toward local group best |
| `GlobalStep` | 0.10 | Pull toward global best |
| `StallLimit` | 3 | (Defined but not yet used in convergence logic) |
| `WaitMagnitude` | 2.0 | Controls patience in `StepUntil` convergence check |

## Code Conventions

- **Minimal dependencies** — The only external dependency is `golang.org/x/sync` (quasi-standard library).
- **Minimization by default** — Fitness functions return lower-is-better values. Maximize by negating.
- **Nil-safe methods** — Optimizer methods check for `nil` receiver and return gracefully.
- **Named returns** — Used in several functions (e.g., `getParticleFitness`, `StepUntil`).
- **Concurrency** — Fitness evaluation uses `golang.org/x/sync/errgroup` with `SetLimit` for bounded parallelism, and `context.Context` for cancellation.
- **No error wrapping** — Errors are simple `fmt.Errorf` sentinel values.
- **Package-level variables** — Sentinel errors are declared as `var` at package scope.
- **Examples are standalone `main` packages** — Each example directory is a separate executable.
- **Embedded data** — The iris example uses `//go:embed` for dataset files.

## Architecture Notes

- The optimizer groups particles into neighborhoods of `LocalSize`. Each particle is influenced by four forces: inertia, its personal best, its local group best, and the global best.
- Bounds are enforced via `Range.Clip()` after position updates and `Range.Contains()` during fitness evaluation (out-of-bounds particles get `nil` fitness).
- Constraints are checked during fitness evaluation — violating particles receive no fitness score and increment their stall counter.
- `StepUntil` uses a logarithmic patience mechanism: it allows more stall steps proportional to the log of total steps taken, controlled by `WaitMagnitude`.
