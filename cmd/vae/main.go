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
var name = flag.String("name", "vae", "name of experiment")
var trainSet = flag.String("train", "mnist_train.csv", "csv for training data")
var seed = flag.Int64("seed", 123456, "Seed for randomness")

func main() {
	flag.Parse()
	rand.Seed(*seed)

	fmt.Println("Loading training set")
	trains, err := mnist.NewSet(*trainSet)
	if err != nil {
		fmt.Println("Error loading training set: ", err)
		return
	}
	fmt.Println("Loading test set")
	encoder := &nn.Network{
		Layers: []nn.Layer{
			&nn.Translate{lab.Solid(28*28, 1, -.5)},
			nn.NewFCLayer(28*28, 200),
			&nn.RELU{},
			nn.NewFCLayer(200, 20),
		},
	}
	reparam := nn.NewReparam(10)
	decoder := &nn.Network{
		Layers: []nn.Layer{
			nn.NewFCLayer(10, 200),
			&nn.RELU{},
			nn.NewFCLayer(200, 28*28),
			&nn.Sigmoid{},
		},
	}
	model := &nn.Network{
		Layers: []nn.Layer{
			encoder,
			reparam,
			decoder,
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
	grid := make([][]*lab.Matrix, 5)
	generated := make([][]*lab.Matrix, 5)
	for i := range grid {
		grid[i] = make([]*lab.Matrix, 6)
		generated[i] = make([]*lab.Matrix, 6)
		for j := range grid[i] {
			grid[i][j] = lab.Gaussian(10, 1)
			generated[i][j] = decoder.Forward(grid[i][j])
			generated[i][j].Cols = 28
			generated[i][j].Rows = 28
		}
	}
	lab.Grid(generated).ImWriteBW(*name + "start.png")
	fmt.Println("Starting Training")

	for i := 0; i < 1000; i++ {
		kl, recon := train(encoder, reparam, decoder, 1, .00001, trains)
		numeral := strconv.FormatInt(int64(i), 10)
		fmt.Println("Epoch ", i+1, " kl loss: ", kl, " recon loss: ", recon)
		model.SaveModel(*name + "E" + numeral + ".json")
		for i := range grid {
			for j := range grid[i] {
				generated[i][j] = decoder.Forward(grid[i][j])
				generated[i][j].Cols = 28
				generated[i][j].Rows = 28
			}
		}
		lab.Grid(generated).ImWriteBW(*name + numeral + "start.png")
	}
}

func train(encoder, reparam, decoder nn.Layer, batchSize int, rate float64, m *mnist.Set) (float64, float64) {
	reconLoss := nn.NewBinaryLogProbLoss(28 * 28)
	klLoss := nn.NewNormalKL(10)
	m.Reset()
	var klTotal float64
	var reconTotal float64
	fmt.Print("Starting...")
	var ctr int
	for {
		ctr++
		fmt.Print("\r", ctr, " training!")
		var kl float64
		var recon float64
		reconLoss.Reset()
		klLoss.Reset()
		var x *lab.Matrix
		for j := 0; j < batchSize; j++ {
			x, _ = m.NextSample()
			if x == nil {
				break
			}
			x = x.Scale(1.0 / 256.0)
			reconLoss.Target = x
			q := encoder.Forward(x)
			kl = klLoss.Loss(q)
			z := reparam.Forward(q)
			xHat := decoder.Forward(z)
			recon = reconLoss.Loss(xHat)
		}
		if x == nil {
			fmt.Println()
			return klTotal, reconTotal
		}
		klTotal += kl
		reconTotal += recon
		encoder.Backward(klLoss.Backward())
		encoder.Backward(reparam.Backward(decoder.Backward(reconLoss.Backward())))
		encoder.Update(rate)
		reparam.Update(rate)
		decoder.Update(rate)
	}

}
