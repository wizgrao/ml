package nn

import (
	"github.com/wizgrao/ml/lab"
	"math"
)

type Layer interface {
	Forward(*lab.Matrix) *lab.Matrix
	Backward(matrix *lab.Matrix) *lab.Matrix
	Update(rate float64)
}

type Loss interface {
	Loss(*lab.Matrix) float64
	Reset()
	Backward() *lab.Matrix
}

type SoftMaxCrossEntropy struct {
	crossEntropy float64
	gradients    *lab.Matrix
	size         int
	Target       int
}

func NewSoftMaxCrossEntropy(size int) *SoftMaxCrossEntropy {
	return &SoftMaxCrossEntropy{
		size:      size,
		gradients: lab.NewMatrix(size, 1),
	}
}

func (s *SoftMaxCrossEntropy) Backward() *lab.Matrix {
	return s.gradients
}

func (s *SoftMaxCrossEntropy) Reset() {
	s.crossEntropy = 0
	s.gradients = lab.NewMatrix(s.size, 1)
}

func (s *SoftMaxCrossEntropy) Loss(mat *lab.Matrix) float64 {
	max := mat.Access(0, 0)
	for i := 1; i < s.size; i++ {
		if mat.Access(i, 0) > max {
			max = mat.Access(i, 0)
		}
	}
	mat = mat.Sub(lab.Solid(s.size, 1, max))
	var denom float64
	for i := 0; i < s.size; i++ {
		denom += math.Exp(mat.Access(i, 0))
	}
	newGradients := lab.NewMatrix(s.size, 1)
	for i := 0; i < s.size; i++ {

		var y float64
		p := math.Exp(mat.Access(i, 0)) / denom
		if i == s.Target {
			y = 1.0
			s.crossEntropy += -math.Log(p)
		}
		newGradients.Set(i, 0, p-y)
	}
	s.gradients = s.gradients.Add(newGradients)
	return s.crossEntropy
}

type FCLayer struct {
	W *lab.Matrix
	B *lab.Matrix

	Wprime      *lab.Matrix
	Bprime      *lab.Matrix
	Input       *lab.Matrix
	Activations *lab.Matrix

	WMomentum *lab.Matrix
	BMomentum *lab.Matrix
}

func NewFCLayer(in, out int) *FCLayer {
	return &FCLayer{
		Wprime:      lab.NewMatrix(out, in),
		Bprime:      lab.NewMatrix(out, 1),
		WMomentum:   lab.NewMatrix(out, in),
		BMomentum:   lab.NewMatrix(out, 1),
		W:           lab.Gaussian(out, in),
		B:           lab.Gaussian(out, 1),
		Input:       lab.NewMatrix(in, 1),
		Activations: lab.NewMatrix(out, 1),
	}
}

func (f *FCLayer) Forward(matrix *lab.Matrix) *lab.Matrix {
	f.Input = matrix
	out := f.W.Multiply(matrix).Add(f.B)
	f.Activations = out
	return out
}

func (f *FCLayer) Backward(matrix *lab.Matrix) *lab.Matrix {
	f.Bprime = matrix
	for i := 0; i < f.Wprime.Rows; i++ {
		for j := 0; j < f.Wprime.Cols; j++ {
			f.Wprime.Set(i, j, f.Input.Access(j, 0)*matrix.Access(i, 0))
		}
	}
	return f.W.Transpose().Multiply(matrix)
}

func (f *FCLayer) Update(rate float64) {
	f.WMomentum = f.WMomentum.Scale(.9).Add(f.Wprime.Scale(rate))
	f.BMomentum = f.BMomentum.Scale(.9).Add(f.Bprime.Scale(rate))
	f.W = f.W.Sub(f.WMomentum)
	f.B = f.B.Sub(f.BMomentum)
}

type TanhActivation struct {
	Input      *lab.Matrix
	Activation *lab.Matrix
}

func (f *TanhActivation) Forward(matrix *lab.Matrix) *lab.Matrix {
	ret := lab.NewMatrix(matrix.Rows, 1)
	f.Input = matrix
	for i, val := range matrix.X {
		ret.X[i] = math.Tanh(val)
	}
	f.Activation = ret
	return ret
}

func (f *TanhActivation) Backward(matrix *lab.Matrix) *lab.Matrix {
	return matrix.MultElems(lab.Solid(matrix.Rows, matrix.Cols, 1).Sub(f.Activation.MultElems(f.Activation)))
}

func (f *TanhActivation) Update(rate float64) {
}

type Scale struct {
	S float64
}

func (f *Scale) Forward(matrix *lab.Matrix) *lab.Matrix {

	return matrix.Scale(f.S)
}

func (f *Scale) Backward(matrix *lab.Matrix) *lab.Matrix {
	return matrix.Scale(f.S)
}

func (f *Scale) Update(rate float64) {
}

type Translate struct {
	V *lab.Matrix
}

func (f *Translate) Forward(matrix *lab.Matrix) *lab.Matrix {

	return matrix.Add(f.V)
}

func (f *Translate) Backward(matrix *lab.Matrix) *lab.Matrix {
	return matrix
}

func (f *Translate) Update(rate float64) {
}

type SELoss struct {
	Target *lab.Matrix
	Diff   *lab.Matrix
	Ct     int
}

func (s *SELoss) Loss(matrix *lab.Matrix) float64 {
	if s.Ct == 0 {
		s.Diff = lab.NewMatrix(matrix.Rows, 1)
	}
	s.Ct++
	s.Diff = s.Diff.Add(matrix.Sub(s.Target))
	return s.Diff.Scale(1/float64(s.Ct)).Transpose().Multiply(s.Diff).Access(0, 0)
}

func (s *SELoss) Reset() {
	s.Ct = 0
}

func (s *SELoss) Backward() *lab.Matrix {
	return s.Diff.Scale(2 * 1 / float64(s.Ct))
}

type Network struct {
	Layers []Layer
}

func (n *Network) Forward(m *lab.Matrix) *lab.Matrix {
	for _, layer := range n.Layers {
		m = layer.Forward(m)
	}
	return m
}

func (n *Network) Backward(m *lab.Matrix) {
	for i := len(n.Layers) - 1; i >= 0; i-- {
		m = n.Layers[i].Backward(m)
	}
}

func (n *Network) Update(rate float64) {
	for _, layer := range n.Layers {
		layer.Update(rate)
	}
}

type RELU struct {
	Input      *lab.Matrix
	Activation *lab.Matrix
}

func (f *RELU) Forward(matrix *lab.Matrix) *lab.Matrix {
	ret := lab.NewMatrix(matrix.Rows, 1)
	f.Input = matrix
	for i, val := range matrix.X {
		ret.X[i] = math.Max(val, .1*val)
	}
	f.Activation = ret
	return ret
}

func (f *RELU) Backward(matrix *lab.Matrix) *lab.Matrix {
	ret := lab.NewMatrix(matrix.Rows, 1)
	f.Input = matrix
	for i, val := range f.Activation.X {
		if val >= 0 {
			ret.X[i] = 1
		} else {
			ret.X[i] = .1
		}
	}
	return ret.MultElems(matrix)
}

func (f *RELU) Update(rate float64) {
}
