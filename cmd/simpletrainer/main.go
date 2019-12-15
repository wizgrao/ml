package main

import (
	"fmt"
	"github.com/wizgrao/ml/lab"
	"github.com/wizgrao/ml/nn"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
)

func main() {
	model := nn.Network{
		Layers: []nn.Layer{
			&nn.Translate{(&lab.Vector{[]float64{-.5, -.5}, 2, 1}).Col()},
			nn.NewFCLayer(2, 6),
			&nn.RELU{},
			nn.NewFCLayer(6, 50),
			&nn.RELU{},
			nn.NewFCLayer(50, 6),
			&nn.RELU{},
			nn.NewFCLayer(6, 2),
		},
	}

	fmt.Println(benchModel(model, 100))

	for i := 0; i < 1000; i++ {
		train(model, 1, 100, .00005)
		fmt.Println(benchModel(model, 1000))
		drawModel(model, fmt.Sprintf("out%v.png", i))
	}

}

func drawModel(model nn.Network, fname string) {
	res := 128
	im := image.NewRGBA(image.Rect(0, 0, res, res))
	red := color.RGBA{255, 0, 0, 255}
	blue := color.RGBA{0, 0, 255, 255}
	for j := 0; j < res; j++ {
		for k := 0; k < res; k++ {
			if getHot(model.Forward(lab.NewVector([]float64{float64(j) / float64(res), float64(k) / float64(res)}).Col())) == 0 {
				im.Set(j, k, blue)
			} else {
				im.Set(j, k, red)
			}
		}
	}
	f, err := os.Create(fname)
	if err != nil {
		fmt.Println("what the", err)
	}
	png.Encode(f, im)
	f.Close()

}

func getCatMat(matrix *lab.Matrix) int {
	center := lab.NewVector([]float64{.5, .5}).Col()
	diff := center.Sub(matrix)
	dist := diff.Transpose().Multiply(diff).Access(0, 0)
	if dist < .15 {
		return 0
	}
	return 1

}

func getHot(matrix *lab.Matrix) int {
	if matrix.X[0] > matrix.X[1] {
		return 0
	}
	return 1
}

func benchModel(network nn.Network, n int) float64 {
	var cor int
	for i := 0; i < n; i++ {
		vec := &lab.Matrix{
			X:    []float64{rand.Float64(), rand.Float64()},
			Cols: 1,
			Rows: 2,
		}
		expected := getCatMat(vec)
		got := getHot(network.Forward(vec))
		if expected == got {
			cor++
		}
	}
	return float64(cor) / float64(n)
}

func train(network nn.Network, batchSize, n int, rate float64) {
	loss := nn.NewSoftMaxCrossEntropy(2)
	for i := 0; i < n; i++ {
		loss.Reset()
		var losss float64
		for j := 0; j < batchSize; j++ {
			vec := &lab.Matrix{
				X:    []float64{rand.Float64(), rand.Float64()},
				Cols: 1,
				Rows: 2,
			}
			loss.Target = getCatMat(vec)
			result := network.Forward(vec)
			losss = loss.Loss(result)
		}
		fmt.Println("loss", losss)
		network.Backward(loss.Backward())
		network.Update(rate)
	}
}
