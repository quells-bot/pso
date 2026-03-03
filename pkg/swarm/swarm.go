package swarm

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"

	"golang.org/x/sync/errgroup"
)

var (
	ErrInvalidShape = fmt.Errorf("shape must have at least 1 dimension")
)

// A Fitness function scores a candidate particle position.
// Lower values have a better score. To maximize a function, return the negative of the value.
// The bool return indicates whether the result is valid. Return false to signal that the
// position is infeasible (e.g. domain error, NaN). The optimizer treats this the same as a
// constraint violation.
type Fitness func([]float64) (float64, bool)

// A Constraint sets hard limits on the range of values each parameter can take relative to other
// parameters. Must return false if a particle position is invalid.
type Constraint func([]float64) bool

type Range [2]float64

// Contains returns true if x is in the range [lower, upper]
func (r Range) Contains(x float64) bool {
	if r[0] == 0 && r[1] == 0 {
		return true
	}

	if x < r[0] || r[1] < x {
		return false
	}

	return true
}

// Clip a value to be in the range [lower, upper]
func (r Range) Clip(x float64) float64 {
	if r[0] == 0 && r[1] == 0 {
		return x
	}

	if x < r[0] {
		return r[0]
	}
	if r[1] < x {
		return r[1]
	}

	return x
}

type Options struct {
	// Size of groups of particles that can "see" one another.
	// Defaults to 25.
	LocalSize uint

	// Number of particles to include in the system.
	// By default, scales with the number of parameters in the Fitness function and the LocalSize.
	PopulationSize uint

	// Number of goroutines to use during optimization.
	// Defaults to GOMAXPROCS.
	Parallelism uint

	// Hard limits on the range of value each parameter can take.
	// Defaults to unbounded for all dimensions.
	// If a bound is the zero value for a Range, that dimension is unbounded.
	Bounds []Range

	// Hard limits on the range of values each parameter can take relative to other parameters.
	// Defaults to unconstrained.
	Constraints []Constraint

	// Hyperparameters which affect how quickly the particles converge on a local minima.

	Inertia      float64 // Defaults to 0.95
	ParticleStep float64 // Defaults to 0.75
	LocalStep    float64 // Defaults to 0.50
	GlobalStep   float64 // Defaults to 0.10
	StallLimit   uint    // Defaults to 3

	// Log progress
	Verbose bool

	WaitMagnitude float64

	groupCount int
}

type Optimizer struct {
	fitness Fitness
	shape   []Range
	options Options
	dims    int

	// Flat contiguous arrays indexed as [idx*dims + d].
	positions  []float64 // [populationSize * dims]
	velocities []float64 // [populationSize * dims]
	stallCount []uint    // [populationSize]

	particleBestPosition []float64 // [populationSize * dims]
	particleBestFitness  []float64 // [populationSize]

	localBestPosition []float64 // [groupCount * dims]
	localBestFitness  []float64 // [groupCount]

	globalBestPosition []float64 // [dims]
	globalBestFitness  float64

	averageFitness float64

	// Pre-allocated buffer reused across steps.
	results []particleFitness
}

func (opt *Optimizer) Best() []float64 {
	if opt == nil {
		return nil
	}

	return opt.globalBestPosition
}

// A New particle swarm optimizer.
//
// The shape is only used to initialize particle positions and velocities.
// It does not impose Constraints or Bounds.
func New(fitness Fitness, shape []Range, options Options) (opt *Optimizer, err error) {
	if len(shape) == 0 {
		err = ErrInvalidShape
		return
	}

	opt = &Optimizer{
		fitness: fitness,
		shape:   shape,
		dims:    len(shape),
	}

	if options.LocalSize == 0 {
		options.LocalSize = 25
	}
	if options.PopulationSize == 0 {
		options.PopulationSize = 10 * options.LocalSize * uint(len(shape))
	}
	options.groupCount = int(options.PopulationSize / options.LocalSize)

	if options.Parallelism == 0 {
		options.Parallelism = uint(runtime.GOMAXPROCS(0))
	}

	if options.Inertia == 0 {
		options.Inertia = 0.95
	}
	if options.ParticleStep == 0 {
		options.ParticleStep = 0.75
	}
	if options.LocalStep == 0 {
		options.LocalStep = 0.5
	}
	if options.GlobalStep == 0 {
		options.GlobalStep = 0.1
	}
	if options.StallLimit == 0 {
		options.StallLimit = 3
	}

	if options.WaitMagnitude == 0.0 {
		options.WaitMagnitude = 2
	}

	opt.options = options

	opt.Reset()
	return
}

func (opt *Optimizer) Reset() {
	if opt == nil {
		return
	}

	popSize := int(opt.options.PopulationSize)
	dims := opt.dims

	opt.positions = make([]float64, popSize*dims)
	opt.velocities = make([]float64, popSize*dims)
	opt.stallCount = make([]uint, popSize)
	opt.particleBestPosition = make([]float64, popSize*dims)
	opt.particleBestFitness = make([]float64, popSize)
	for i := range popSize {
		opt.particleBestFitness[i] = math.MaxFloat64
	}

	for d, r := range opt.shape {
		delta := r[1] - r[0]
		for i := range popSize {
			off := i*dims + d
			opt.positions[off] = r[0] + delta*rand.Float64()
			opt.velocities[off] = (2*rand.Float64() - 1) * delta
		}
	}
	copy(opt.particleBestPosition, opt.positions)

	groupCount := opt.options.groupCount
	opt.localBestPosition = make([]float64, groupCount*dims)
	opt.localBestFitness = make([]float64, groupCount)
	for i := range groupCount {
		opt.localBestFitness[i] = math.MaxFloat64
		copy(opt.localBestPosition[i*dims:(i+1)*dims], opt.positions[i*int(opt.options.LocalSize)*dims:(i*int(opt.options.LocalSize)+1)*dims])
	}

	opt.globalBestPosition = make([]float64, dims)
	copy(opt.globalBestPosition, opt.positions[:dims])
	opt.globalBestFitness = math.MaxFloat64

	opt.results = make([]particleFitness, popSize)
}

type particleFitness struct {
	idx     int
	fitness float64
	valid   bool
}

// calculate fitness of particle at idx
func (opt *Optimizer) getParticleFitness(idx int) (result particleFitness) {
	if opt == nil {
		return
	}

	result.idx = idx

	position := opt.positions[idx*opt.dims : (idx+1)*opt.dims]
	for i, bounds := range opt.options.Bounds {
		if !bounds.Contains(position[i]) {
			// position out of bounds
			return
		}
	}
	for _, withinConstraint := range opt.options.Constraints {
		if !withinConstraint(position) {
			// position exceeds constraint
			return
		}
	}

	fitness, ok := opt.fitness(position)
	if !ok {
		return
	}
	result.fitness = fitness
	result.valid = true
	return
}

func (opt *Optimizer) updateFitness(ctx context.Context) error {
	if opt == nil {
		return nil
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	popSize := int(opt.options.PopulationSize)
	dims := opt.dims

	// Zero the pre-allocated results buffer.
	for i := range popSize {
		opt.results[i] = particleFitness{}
	}

	var g errgroup.Group
	g.SetLimit(int(opt.options.Parallelism))

	for idx := range popSize {
		g.Go(func() error {
			if err := ctx.Err(); err != nil {
				return err
			}
			opt.results[idx] = opt.getParticleFitness(idx)
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	// Process results sequentially
	var avg, count float64
	for _, result := range opt.results[:popSize] {
		if !result.valid {
			opt.stallCount[result.idx]++
			continue
		}

		avg += result.fitness
		count++

		if result.fitness < opt.particleBestFitness[result.idx] {
			opt.particleBestFitness[result.idx] = result.fitness
			pOff := result.idx * dims
			copy(opt.particleBestPosition[pOff:pOff+dims], opt.positions[pOff:pOff+dims])
		}

		groupIdx := result.idx % opt.options.groupCount
		if result.fitness < opt.localBestFitness[groupIdx] {
			opt.localBestFitness[groupIdx] = result.fitness
			gOff := groupIdx * dims
			pOff := result.idx * dims
			copy(opt.localBestPosition[gOff:gOff+dims], opt.positions[pOff:pOff+dims])
		}

		if result.fitness < opt.globalBestFitness {
			opt.globalBestFitness = result.fitness
			pOff := result.idx * dims
			copy(opt.globalBestPosition, opt.positions[pOff:pOff+dims])
		}
	}
	if count > 0 {
		opt.averageFitness = avg / count
	}

	return nil
}

func (opt *Optimizer) Step(ctx context.Context) error {
	if err := opt.updateFitness(ctx); err != nil {
		return err
	}

	popSize := int(opt.options.PopulationSize)
	dims := opt.dims
	nBounds := len(opt.options.Bounds)

	for idx := range popSize {
		groupIdx := idx % opt.options.groupCount
		off := idx * dims
		gOff := groupIdx * dims

		for d := range dims {
			ri := opt.positions[off+d]
			rp := (opt.particleBestPosition[off+d] - ri) * rand.Float64()
			rl := (opt.localBestPosition[gOff+d] - ri) * rand.Float64()
			rg := (opt.globalBestPosition[d] - ri) * rand.Float64()

			nv := opt.velocities[off+d]*opt.options.Inertia +
				rp*opt.options.ParticleStep +
				rl*opt.options.LocalStep +
				rg*opt.options.GlobalStep
			opt.velocities[off+d] = nv

			np := ri + nv
			if d < nBounds {
				np = opt.options.Bounds[d].Clip(np)
			}
			opt.positions[off+d] = np
		}
	}

	return nil
}

func (opt *Optimizer) StepUntil(ctx context.Context, progressRate float64) (steps int, err error) {
	if opt == nil {
		return
	}

	if err = opt.Step(ctx); err != nil {
		return
	}
	steps = 1

	minProgressRate := math.Abs(progressRate)
	last := opt.globalBestFitness
	stepsSinceImprovement := 0

	for {
		if err = ctx.Err(); err != nil {
			return
		}

		if err = opt.Step(ctx); err != nil {
			return
		}
		steps++
		if opt.options.Verbose {
			log.Println(steps, opt.globalBestFitness, opt.averageFitness)
		}

		if last-opt.globalBestFitness < minProgressRate {
			stepsSinceImprovement++

			waitLimit := math.Log10(float64(steps))
			waitedFor := math.Log10(float64(stepsSinceImprovement))
			if waitedFor > opt.options.WaitMagnitude && waitLimit-waitedFor < opt.options.WaitMagnitude {
				break
			}
		} else {
			if opt.options.Verbose {
				log.Println(steps, opt.globalBestFitness)
			}
			stepsSinceImprovement = 0
		}

		last = opt.globalBestFitness
	}

	if opt.options.Verbose {
		log.Println(steps, opt.globalBestFitness)
	}
	return steps, nil
}
