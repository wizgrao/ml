package main

import (
	"fmt"
	"github.com/wizgrao/ml/lab"
	"github.com/wizgrao/ml/nn"

	"math"
	"math/rand"
)

func main() {

	trainSet, err := newMNIST("mnist_train.csv")
	if err != nil {
		fmt.Println(err)
		return
	}
	testSet, err := newMNIST("mnist_test.csv")
	if err != nil {
		fmt.Println(err)
		return
	}
	model := nn.Network{
		Layers: []nn.Layer{
			&nn.Translate{lab.Solid(28*28, 1, -128.0)},
			&nn.Scale{1.0 / 128.0},
			nn.NewFCLayer(28*28, 500),
			&nn.RELU{},
			nn.NewFCLayer(500, 10),
		},
	}
	fmt.Println(evaluate(model, testSet))
	for i := 0; i < 1000; i++ {
		train(model, 10, .00001, trainSet)
		fmt.Println(evaluate(model, trainSet), evaluate(model, testSet))
	}
}

func train(network nn.Network, batchSize int, rate float64, m *mnist) {
	loss := nn.NewSoftMaxCrossEntropy(10)
	m.reset()
	for {
		loss.Reset()
		var x *lab.Matrix
		var target int
		for j := 0; j < batchSize; j++ {
			x, target = m.nextSample()
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

func evaluate(network nn.Network, m *mnist) float64 {
	m.reset()
	var correct int
	var total int
	confusion := lab.NewMatrix(10, 10)
	for i := 0; i < 500; i++ {
		x, t := m.nextSample()
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
		return 1
	}
	fmt.Println(confusion)
	return float64(correct) / float64(total)
}

type mnist struct {
	i    int
	mat  *lab.Matrix
	perm []int
}

func newMNIST(fileName string) (*mnist, error) {
	mat, err := lab.LoadCSV(fileName)
	if err != nil {
		return nil, err
	}
	mat = mat.Transpose()
	return &mnist{
		mat:  mat,
		perm: rand.Perm(mat.Cols),
	}, nil
}

func (m *mnist) nextSample() (*lab.Matrix, int) {
	if m.i >= m.mat.Cols {
		return nil, 0
	}
	index := m.perm[m.i]
	label := int(math.Round(m.mat.Access(0, index)))
	x := m.mat.SubMatrix(1, index, 28*28, 1)
	m.i++
	return x, label
}

func (m *mnist) reset() {
	m.i = 0
	m.perm = rand.Perm(m.mat.Cols)
}
