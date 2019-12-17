package lab

import (
	"encoding/csv"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

type Matrix struct {
	X    []float64 //row * cols + col
	Cols int
	Rows int
}

func (m *Matrix) Exp() *Matrix {
	r := NewMatrix(m.Rows, m.Cols)
	for i, x := range r.X {
		r.X[i] = math.Exp(x)
	}
	return r
}

func VStack(matrices ...*Matrix) *Matrix {
	if len(matrices) == 0 {
		return nil
	}
	var x []float64
	for _, matrix := range matrices {
		x = append(x, matrix.X...)
	}
	return &Matrix{
		X:    x,
		Cols: matrices[0].Cols,
		Rows: len(x) / matrices[0].Cols,
	}
}

func HStack(matrices ...*Matrix) *Matrix {
	newMatrices := make([]*Matrix, len(matrices))
	for i, matrix := range matrices {
		newMatrices[i] = matrix.Transpose()
	}
	return VStack(newMatrices...).Transpose()
}

func Grid(matrices [][]*Matrix) *Matrix {
	newMatrices := make([]*Matrix, len(matrices))
	for i, row := range matrices {
		newMatrices[i] = HStack(row...)
	}
	return VStack(newMatrices...)
}

func (m *Matrix) ImWriteBW(fname string) error {
	image := image.NewNRGBA(image.Rect(0, 0, m.Cols, m.Rows))
	max := m.X[0]
	min := m.X[0]

	for _, v := range m.X {
		if v > max {
			max = v
		}
		if v < min {
			min = v
		}
	}
	for x := 0; x < m.Cols; x++ {
		for y := 0; y < m.Rows; y++ {
			c := uint8(255.0 / (max - min) * (m.Access(y, x) - min))
			image.Set(x, y, color.NRGBA{
				R: c,
				G: c,
				B: c,
				A: 255,
			})
		}
	}
	f, err := os.Create(fname)
	defer f.Close()
	if err != nil {
		return err
	}
	return png.Encode(f, image)

}

func LoadCSV(fileName string) (*Matrix, error) {
	var buffer []float64
	var rows int
	file, err := os.Open(fileName)
	defer file.Close()
	if err != nil {
		return nil, err
	}

	r := csv.NewReader(file)
	var line []float64

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if rows == 0 {
			line = make([]float64, len(record))
		}
		rows++
		for i, val := range record {
			line[i], err = strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, err
			}
		}
		buffer = append(buffer, line...)
	}
	cols := len(buffer) / rows
	return &Matrix{
		X:    buffer,
		Rows: rows,
		Cols: cols,
	}, nil
}

func (m *Matrix) Multiply(m1 *Matrix) *Matrix {
	if m.Cols != m1.Rows {
		panic("fucc.go dude ur dimensions r fucced")
	}
	mout := NewMatrix(m.Rows, m1.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m1.Cols; j++ {
			mout.X[mout.Cols*i+j] = m.Row(i).Dot(m1.Col(j))
		}
	}
	return mout
}

func (m *Matrix) Add(m1 *Matrix) *Matrix {
	if m.Cols != m1.Cols || m.Rows != m1.Rows {
		panic("bounds don't match")
	}
	ret := NewMatrix(m.Rows, m.Cols)
	for i := range m.X {
		ret.X[i] = m.X[i] + m1.X[i]
	}
	return ret
}

func (m *Matrix) MultElems(m1 *Matrix) *Matrix {
	if m.Cols != m1.Cols || m.Rows != m1.Rows {
		panic("bounds don't match")
	}
	ret := NewMatrix(m.Rows, m.Cols)
	for i := range m.X {
		ret.X[i] = m.X[i] * m1.X[i]
	}
	return ret
}

func (m *Matrix) SubMatrix(i, j, rows, columns int) *Matrix {

	ret := NewMatrix(rows, columns)
	for ii := 0; ii < rows; ii++ {
		for jj := 0; jj < columns; jj++ {
			ret.Set(ii, jj, m.Access(i+ii, j+jj))
		}
	}

	return ret
}

func (m *Matrix) Sub(m1 *Matrix) *Matrix {
	if m.Cols != m1.Cols || m.Rows != m1.Rows {
		panic("bounds don't match")
	}
	ret := NewMatrix(m.Rows, m.Cols)
	for i := range m.X {
		ret.X[i] = m.X[i] - m1.X[i]
	}
	return ret
}

func (m *Matrix) Scale(x float64) *Matrix {
	ret := NewMatrix(m.Rows, m.Cols)
	for i := range m.X {
		ret.X[i] = m.X[i] * x
	}
	return ret
}

func (m *Matrix) Access(i, j int) float64 {
	if i < 0 || i > m.Rows || j < 0 || j > m.Cols {
		panic("fucc.go bruv get ur indices right")
	}
	return m.X[m.Cols*i+j]
}

func (m *Matrix) Set(i, j int, a float64) {
	if i < 0 || i > m.Rows || j < 0 || j > m.Cols {
		panic("fucc.go bruv get ur indices right")
	}
	m.X[m.Cols*i+j] = a
}

func (m *Matrix) Row(i int) *Vector {
	return &Vector{
		X:    m.X[i*m.Cols : (i+1)*m.Cols],
		Size: m.Cols,
		Skip: 1,
	}
}

func (m *Matrix) Col(i int) *Vector {
	return &Vector{
		X:    m.X[i : (m.Rows-1)*m.Cols+i+1],
		Size: m.Rows,
		Skip: m.Cols,
	}
}

func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		X:    make([]float64, rows*cols, rows*cols),
		Cols: cols,
		Rows: rows,
	}
}

func Gaussian(rows, cols int) *Matrix {
	mat := NewMatrix(rows, cols)
	for i := range mat.X {
		mat.X[i] = rand.NormFloat64()
	}
	return mat
}

func Solid(rows, cols int, val float64) *Matrix {
	mat := NewMatrix(rows, cols)
	for i := range mat.X {
		mat.X[i] = val
	}
	return mat
}

func IdMatrix(n int) *Matrix {
	m := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		m.X[i+i*n] = 1
	}
	return m
}

type Vector struct {
	X    []float64
	Size int
	Skip int
}

func (v *Vector) Access(a int) float64 {
	return v.X[a*v.Skip]
}

func (v *Vector) Set(a int, f float64) {
	v.X[a*v.Skip] = f
}

func (v *Vector) SetV(v1 *Vector) {
	if v.Size != v1.Size {
		panic("bro y ur vector sizes different")
	}
	for i := 0; i < v.Size; i++ {
		v.X[i*v.Skip] = v1.X[i*v1.Skip]
	}
}

func (v *Vector) Dot(v1 *Vector) float64 {
	var ans float64
	for i := 0; i < v.Size; i++ {
		ans += v.X[i*v.Skip] * v1.X[i*v1.Skip]
	}
	return ans
}

func NewVector(f []float64) *Vector {
	return &Vector{
		X:    f,
		Size: len(f),
		Skip: 1,
	}
}

func (v *Vector) Col() *Matrix {
	mat := NewMatrix(v.Size, 1)
	mat.Col(0).SetV(v)
	return mat
}
func (v *Vector) Row() *Matrix {
	mat := NewMatrix(1, v.Size)
	mat.Row(0).SetV(v)
	return mat
}
func (v *Vector) String() string {
	var nums []string
	for i := 0; i < v.Size; i++ {
		nums = append(nums, fmt.Sprintf("%f", v.Access(i)))
	}
	return "[" + strings.Join(nums, " ") + "]"
}

func (m *Matrix) String() string {
	var nums []string
	for i := 0; i < m.Rows; i++ {
		nums = append(nums, m.Row(i).String())
	}
	return "[" + strings.Join(nums, "\n ") + "]"
}

func (m *Matrix) Transpose() *Matrix {
	mat := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			mat.X[j*m.Rows+i] = m.X[i*m.Cols+j]
		}
	}
	return mat
}
