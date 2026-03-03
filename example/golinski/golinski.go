package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/quells/pso/pkg/swarm"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	shape := []swarm.Range{
		{2.6, 3.6},
		{0.7, 0.8},
		{17, 28},
		{7.3, 8.3},
		{7.3, 8.3},
		{2.9, 3.9},
		{5.0, 5.5},
	}

	constraints := []swarm.Constraint{
		func(x []float64) bool { return true },
		func(x []float64) bool { return 27.0/(x[0]*math.Pow(x[1], 2)*x[2]) <= 1 },
		func(x []float64) bool { return 397.5/(x[0]*math.Pow(x[1], 2)*math.Pow(x[2], 2)) <= 1 },
		func(x []float64) bool { return 1.93*math.Pow(x[3], 3)/(x[1]*x[2]*math.Pow(x[5], 4)) <= 1 },
		func(x []float64) bool { return 1.93*math.Pow(x[4], 3)/(x[1]*x[2]*math.Pow(x[6], 4)) <= 1 },
		func(x []float64) bool {
			return math.Sqrt(math.Pow(745*x[3]/x[1]/x[2], 2)+16.9*1e6)/(110*math.Pow(x[5], 3)) <= 1
		},
		func(x []float64) bool {
			return math.Sqrt(math.Pow(745*x[4]/x[1]/x[2], 2)+157.5*1e6)/(85*math.Pow(x[6], 3)) <= 1
		},
		func(x []float64) bool { return x[1]*x[2]/40 <= 1 },
		func(x []float64) bool { return 5*x[1]/x[0] <= 1 },
		func(x []float64) bool { return x[0]/12/x[1] <= 1 },
		func(x []float64) bool { return (1.5*x[5]+1.9)/x[3] <= 1 },
		func(x []float64) bool { return (1.1*x[6]+1.9)/x[4] <= 1 },
	}

	options := swarm.Options{
		Bounds:      shape,
		Constraints: constraints,

		PopulationSize: 14000,
		WaitMagnitude:  2.5,
	}

	pso, err := swarm.New(golinski, shape, options)
	if err != nil {
		log.Fatalf("could not build swarm: %v", err)
	}

	start := time.Now()
	numSteps, err := pso.StepUntil(context.Background(), 1e-6)
	if err != nil {
		log.Fatalf("optimization failed: %v", err)
	}
	elapsed := time.Since(start)
	nsPerStep := elapsed.Nanoseconds() / int64(numSteps)
	perStep := time.Duration(nsPerStep)
	fmt.Printf("%d steps @ %s per step\n", numSteps, perStep)

	best := pso.Best()
	fmt.Println(best, golinski(best))

	knownBest := []float64{3.50, 0.7, 17, 7.3, 7.30, 3.35, 5.29}
	fmt.Println(knownBest, golinski(knownBest))
}

func golinski(x []float64) float64 {
	a := 0.7854 *
		x[0] *
		math.Pow(x[1], 2) *
		(3.3333*math.Pow(x[2], 2) + 14.9334*x[2] - 43.0934)

	b := 1.508 *
		x[0] *
		(math.Pow(x[5], 2) + math.Pow(x[6], 2))

	c := 7.4777 * (math.Pow(x[5], 3) + math.Pow(x[6], 3))

	d := 0.7854 * (x[3]*math.Pow(x[5], 2) + x[4]*math.Pow(x[6], 2))

	return a - b + c + d
}
