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
type Fitness func([]float64) float64

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

	positions  [][]float64 // [populationSize]position
	velocities [][]float64 // [populationSize]velocity
	stallCount []uint

	particleBestPosition [][]float64 // [populationSize]position
	particleBestFitness  []float64   // [populationSize]fitness

	localBestPosition [][]float64 // [groupCount]position
	localBestFitness  []float64   // [groupCount]fitness

	globalBestPosition []float64 // position
	globalBestFitness  float64

	averageFitness float64
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

	opt.positions = make([][]float64, opt.options.PopulationSize)
	opt.velocities = make([][]float64, opt.options.PopulationSize)
	opt.stallCount = make([]uint, opt.options.PopulationSize)
	opt.particleBestPosition = make([][]float64, opt.options.PopulationSize)
	opt.particleBestFitness = make([]float64, opt.options.PopulationSize)
	for i := 0; i < int(opt.options.PopulationSize); i++ {
		opt.positions[i] = make([]float64, len(opt.shape))
		opt.velocities[i] = make([]float64, len(opt.shape))
		opt.particleBestPosition[i] = make([]float64, len(opt.shape))
		opt.particleBestFitness[i] = math.MaxFloat64
	}
	for j, r := range opt.shape {
		for i := 0; i < int(opt.options.PopulationSize); i++ {
			delta := r[1] - r[0]
			opt.positions[i][j] = r[0] + delta*rand.Float64()
			opt.velocities[i][j] = (2*rand.Float64() - 1) * delta
		}
	}
	copy(opt.particleBestPosition, opt.positions)

	opt.localBestPosition = make([][]float64, opt.options.groupCount)
	opt.localBestFitness = make([]float64, opt.options.groupCount)
	for i := 0; i < opt.options.groupCount; i++ {
		opt.localBestPosition[i] = make([]float64, len(opt.shape))
		idx := i * int(opt.options.LocalSize)
		copy(opt.localBestPosition[i], opt.positions[idx])
		opt.localBestFitness[i] = math.MaxFloat64
	}

	opt.globalBestPosition = make([]float64, len(opt.shape))
	copy(opt.globalBestPosition, opt.positions[0])
	opt.globalBestFitness = math.MaxFloat64
}

type particleFitness struct {
	idx     int
	fitness *float64
}

// calculate fitness of particle at idx
func (opt *Optimizer) getParticleFitness(idx int) (result particleFitness) {
	if opt == nil {
		return
	}

	result.idx = idx

	position := opt.positions[idx]
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

	fitness := opt.fitness(position)
	result.fitness = &fitness
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
	results := make([]particleFitness, popSize)

	var g errgroup.Group
	g.SetLimit(int(opt.options.Parallelism))

	for idx := range popSize {
		g.Go(func() error {
			if err := ctx.Err(); err != nil {
				return err
			}
			results[idx] = opt.getParticleFitness(idx)
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	// Process results sequentially
	var avg, count float64
	for _, result := range results {
		if result.fitness == nil {
			opt.stallCount[result.idx]++
			continue
		}

		avg += *result.fitness
		count++

		if *result.fitness < opt.particleBestFitness[result.idx] {
			opt.particleBestFitness[result.idx] = *result.fitness
			copy(opt.particleBestPosition[result.idx], opt.positions[result.idx])
		}

		groupIdx := result.idx % opt.options.groupCount
		if *result.fitness < opt.localBestFitness[groupIdx] {
			opt.localBestFitness[groupIdx] = *result.fitness
			copy(opt.localBestPosition[groupIdx], opt.positions[result.idx])
		}

		if *result.fitness < opt.globalBestFitness {
			opt.globalBestFitness = *result.fitness
			copy(opt.globalBestPosition, opt.positions[result.idx])
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

	for idx := 0; idx < int(opt.options.PopulationSize); idx++ {
		groupIdx := idx % opt.options.groupCount

		ri := opt.positions[idx]
		rp := scale(sub(opt.particleBestPosition[idx], ri), rand.Float64())
		rl := scale(sub(opt.localBestPosition[groupIdx], ri), rand.Float64())
		rg := scale(sub(opt.globalBestPosition, ri), rand.Float64())
		vi := opt.velocities[idx]

		nv := sum(
			scale(vi, opt.options.Inertia),
			scale(rp, opt.options.ParticleStep),
			scale(rl, opt.options.LocalStep),
			scale(rg, opt.options.GlobalStep),
		)
		opt.velocities[idx] = nv

		np := sum(opt.positions[idx], nv)
		for i, bounds := range opt.options.Bounds {
			opt.positions[idx][i] = bounds.Clip(np[i])
		}
	}

	return nil
}

// element-wise vector summation, yields v0 + v1 + ... + vi
func sum(vs ...[]float64) []float64 {
	result := make([]float64, len(vs[0]))
	for _, v := range vs {
		for i, vi := range v {
			result[i] += vi
		}
	}
	return result
}

// element-wise vector subtraction, yields u - v
func sub(u, v []float64) []float64 {
	result := make([]float64, len(u))
	for i, ui := range u {
		result[i] = ui - v[i]
	}
	return result
}

// scale each element of a vector by a scalar
func scale(v []float64, s float64) []float64 {
	result := make([]float64, len(v))
	for i, vi := range v {
		result[i] = vi * s
	}
	return result
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
