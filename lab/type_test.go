package lab

import (
	"fmt"
	"testing"
)

func TestIdMatrix(t *testing.T) {
	m := IdMatrix(4)
	fmt.Println(m)

	fmt.Println(m.Col(2), m.Col(2).Dot(m.Col(2)))
	x := m.Scale(2).Multiply(m.Col(2).Col())
	fmt.Println(x)
	newMat := NewMatrix(2, 2)
	newMat.X = []float64{
		1, 2,
		3, 4,
	}
	vec := NewMatrix(2, 1)
	vec.X = []float64{
		1,
		2,
	}
	vec2 := NewVector([]float64{3, 4}).Row()
	fmt.Println(newMat)
	fmt.Println(newMat.Transpose())
	fmt.Println(newMat.Multiply(newMat))
	fmt.Println(newMat.Multiply(vec))
	fmt.Println(vec2.Multiply(vec))
}
