package main

import (
	"github.com/wizgrao/ml/data/mnist"
	"github.com/wizgrao/ml/lab"
	"github.com/wizgrao/ml/nn"

	"flag"
	"fmt"
	"math/rand"
	"strconv"
)

var loadWeights = flag.String("weights", "", "file to load weights from")
var trainSet = flag.String("train", "mnist_train.csv", "csv for training data")
var testSet = flag.String("test", "mnist_test.csv", "csv for test data")
var seed = flag.Int64("seed", 123456, "Seed for randomness")

func main() {
	flag.Parse()
	rand.Seed(*seed)

	fmt.Println("Loading training set")
	trainSet, err := mnist.NewSet(*trainSet)
	if err != nil {
		fmt.Println("Error loading training set: ", err)
		return
	}
	fmt.Println("Loading test set")
	testSet, err := mnist.NewSet(*testSet)
	if err != nil {
		fmt.Println("Error loading test set: ", err)
		return
	}
	model := &nn.Network{
		Layers: []nn.Layer{
			&nn.Translate{lab.Solid(28*28, 1, -128.0)},
			&nn.Scale{1.0 / 128.0},
			nn.NewFCLayer(28*28, 100),
			&nn.RELU{},
			nn.NewFCLayer(100, 10),
		},
	}
	if *loadWeights != "" {
		fmt.Println("Loading weights")
		err := model.LoadModel(*loadWeights)
		if err != nil {
			fmt.Println("Error loading model: ", err)
			return
		}
	}
	s, i := trainSet.NextSample()
	s.Cols = 28
	s.Rows = 28
	grid := make([][]*lab.Matrix, 5)
	for i := range grid {
		grid[i] = make([]*lab.Matrix, 6)
		for j := range grid[i] {
			grid[i][j], _ = trainSet.NextSample()
			grid[i][j].Rows = 28
			grid[i][j].Cols = 28
		}
	}
	fmt.Println("Displaying ", i)
	lab.Grid(grid).ImWriteBW("asdf.png")
	fmt.Println("Starting Training")

	for i := 0; i < 1000; i++ {
		train(model, 10, .00001, trainSet)
		trainAccuracy, trainConfusion := evaluate(model, trainSet)
		testAccuracy, testConfusion := evaluate(model, testSet)
		numeral := strconv.FormatInt(int64(i), 10)
		trainConfusion.ImWriteBW("trainConfusion" + numeral + ".png")
		testConfusion.ImWriteBW("testConfusion" + numeral + ".png")
		fmt.Println("Epoch ", i+1, " training accuracy: ", trainAccuracy, " test accuracy: ", testAccuracy)
		model.SaveModel("mnistE" + numeral + ".json")
	}
}

func train(network *nn.Network, batchSize int, rate float64, m *mnist.Set) {
	loss := nn.NewSoftMaxCrossEntropy(10)
	m.Reset()
	for {
		loss.Reset()
		var x *lab.Matrix
		var target int
		for j := 0; j < batchSize; j++ {
			x, target = m.NextSample()
			if x == nil {
				break
			}
			loss.Target = target
			result := network.Forward(x)
			loss.Loss(result)
		}
		if x == nil {
			return
		}
		network.Backward(loss.Backward())
		network.Update(rate)
	}
}

func evaluate(network *nn.Network, m *mnist.Set) (float64, *lab.Matrix) {
	m.Reset()
	var correct int
	var total int
	confusion := lab.NewMatrix(10, 10)
	for i := 0; i < 500; i++ {
		x, t := m.NextSample()
		if x == nil {
			break
		}
		max := 0
		result := network.Forward(x)
		for i := 1; i < 10; i++ {
			if result.Access(i, 0) > result.Access(max, 0) {
				max = i
			}
		}
		if max == t {
			correct++
		}
		confusion.Set(max, t, confusion.Access(max, t)+1.0)
		total++
	}
	if total == 0 {
		return 1, nil
	}
	return float64(correct) / float64(total), confusion
}
