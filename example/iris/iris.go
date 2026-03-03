package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"time"

	"github.com/quells/pso/pkg/swarm"
)

const (
	numEdges = 55
)

func main() {
	rand.Seed(time.Now().UnixNano())

	shape := make([]swarm.Range, numEdges)
	for i := 0; i < numEdges; i++ {
		shape[i][0] = -10.0
		shape[i][1] = 10.0
	}

	options := swarm.Options{
		PopulationSize: numEdges * 2,
		LocalSize:      2,
		WaitMagnitude:  2.5,
	}

	pso, err := swarm.New(train, shape, options)
	if err != nil {
		log.Fatalf("could not build swarm: %v", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	_, err = pso.StepUntil(ctx, 1e-6)

	best := pso.Best()
	fmt.Println(best)
	fmt.Println(test(best))

	if err != nil {
		fmt.Println("interrupted")
		os.Exit(1)
	}
}

func train(weights []float64) (score float64) {
	for _, flower := range trainingData {
		predicted := feedForwardNN(weights, flower.values)[flower.label]
		score -= predicted
	}
	return
}

func test(weights []float64) (score float64) {
	for _, flower := range trainingData {
		predicted := argmax(feedForwardNN(weights, flower.values))
		if predicted == flower.label {
			score++
		}
	}
	score /= float64(len(trainingData))
	return
}
