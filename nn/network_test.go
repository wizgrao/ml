package nn

import (
	"fmt"
	"github.com/wizgrao/ml/lab"
	"testing"
)

func TestNewFCLayer(t *testing.T) {
	a := NewFCLayer(1, 1)
	fmt.Println(a.W)
	fmt.Println(a.B)
	fmt.Println(lab.Gaussian(2, 2))
}
