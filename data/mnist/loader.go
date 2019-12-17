package mnist

import (
	"github.com/wizgrao/ml/lab"
	"math"
	"math/rand"
)

type Set struct {
	i    int
	mat  *lab.Matrix
	perm []int
}

func NewSet(fileName string) (*Set, error) {
	mat, err := lab.LoadCSV(fileName)
	if err != nil {
		return nil, err
	}
	mat = mat.Transpose()
	return &Set{
		mat:  mat,
		perm: rand.Perm(mat.Cols),
	}, nil
}

func (m *Set) NextSample() (*lab.Matrix, int) {
	if m.i >= m.mat.Cols {
		return nil, 0
	}
	index := m.perm[m.i]
	label := int(math.Round(m.mat.Access(0, index)))
	x := m.mat.SubMatrix(1, index, 28*28, 1)
	m.i++
	return x, label
}

func (m *Set) Reset() {
	m.i = 0
	m.perm = rand.Perm(m.mat.Cols)
}
