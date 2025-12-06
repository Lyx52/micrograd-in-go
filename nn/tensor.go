package nn

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
)

type TensorValue interface {
	*Tensor | float64
}
type Tensor struct {
	Backing   *NArray
	Gradients []float64
	Children  []*Tensor
	backward  func(parent *Tensor)
}

func NewTensorFromArray(values []float64) *Tensor {
	return &Tensor{
		Backing:   NewNArray(values),
		Gradients: make([]float64, len(values)),
		Children:  make([]*Tensor, 0),
		backward:  nil,
	}
}

func NewTensor(values ...float64) *Tensor {
	return NewTensorFromArray(values)
}

func NewTensorRand(length int, rand *rand.Rand) *Tensor {
	return &Tensor{
		Backing:   NewNArrayRand(length, rand),
		Gradients: make([]float64, length),
		Children:  make([]*Tensor, 0),
		backward:  nil,
	}
}

func NewTensorEmpty(length int) *Tensor {
	return &Tensor{
		Backing:   NewNArrayEmpty(length),
		Gradients: make([]float64, length),
		Children:  make([]*Tensor, 0),
		backward:  nil,
	}
}

func NewTensorWithValue(length int, value float64) *Tensor {
	return &Tensor{
		Backing:   NewNArrayFilledWith(length, value),
		Gradients: make([]float64, length),
		Children:  make([]*Tensor, 0),
		backward:  nil,
	}
}

func NewFromTensors(tensors []*Tensor) *Tensor {
	backing := make([]float64, 0)
	gradients := make([]float64, 0)
	for _, tensor := range tensors {
		backing = append(backing, tensor.Backing.Backing...)
		gradients = append(gradients, tensor.Gradients...)
	}

	return &Tensor{
		Backing:   NewNArray(backing),
		Gradients: gradients,
		Children:  tensors,
		backward:  FromTensorsBackward,
	}
}

func DivBackward(parent *Tensor) {
	epsilon := 1e-7
	first := parent.Children[0]
	second := parent.Children[1]
	x := first.Backing
	y := MapFloat64Slice(second.Backing.Backing, func(value float64, i int) float64 {
		return math.Copysign(math.Max(epsilon, math.Abs(value)), value)
	})

	if parent.IsScalar() {
		for i, _ := range first.Gradients {
			first.Gradients[i] += (1 / y[i]) * parent.Gradients[0]
			second.Gradients[i] += (-x.Backing[i] / (y[i] * y[i])) * parent.Gradients[0]
		}
	} else {
		if !parent.TensorEqual(first) || !parent.TensorEqual(second) {
			panic("DivBackward: Expected child tensor to be equal to parent")
		}

		for i, _ := range first.Gradients {
			first.Gradients[i] += (1 / y[i]) * parent.Gradients[i]
			second.Gradients[i] += (-x.Backing[i] / (y[i] * y[i])) * parent.Gradients[i]
		}
	}
}

func SumBackward(parent *Tensor) {
	AddBackward(parent)
}

func AddBackward(parent *Tensor) {
	if parent.IsScalar() {
		for _, child := range parent.Children {
			for i, _ := range child.Gradients {
				child.Gradients[i] += parent.Gradients[0]
			}
		}
	} else {

		for _, child := range parent.Children {
			if !parent.TensorEqual(child) {
				panic("AddBackward: Expected child tensor to be equal to parent")
			}

			for i, _ := range child.Gradients {
				child.Gradients[i] += parent.Gradients[i]
			}
		}
	}
}

func MulBackward(parent *Tensor) {
	first := parent.Children[0]
	second := parent.Children[1]

	if parent.IsScalar() {
		for i, _ := range first.Gradients {
			first.Gradients[i] += second.Backing.Backing[i] * parent.Gradients[0]
			second.Gradients[i] += first.Backing.Backing[i] * parent.Gradients[0]
		}
	} else {
		if !parent.TensorEqual(first) || !parent.TensorEqual(second) {
			panic("MulBackward: Expected child tensor to be equal to parent")
		}

		for i, _ := range first.Gradients {
			first.Gradients[i] += second.Backing.Backing[i] * parent.Gradients[i]
			second.Gradients[i] += first.Backing.Backing[i] * parent.Gradients[i]
		}
	}
}

func LogBackward(parent *Tensor) {
	epsilon := 1e-7
	child := parent.Children[0]
	if parent.IsScalar() {
		for i, _ := range child.Gradients {
			child.Gradients[i] += (1.0 / math.Max(epsilon, child.Backing.Backing[i])) * parent.Gradients[0]
		}
	} else {
		if !parent.TensorEqual(child) {
			panic("LogBackward: Expected child tensor to be equal to parent")
		}

		for i, _ := range child.Gradients {
			child.Gradients[i] += (1.0 / math.Max(epsilon, child.Backing.Backing[i])) * parent.Gradients[i]
		}
	}
}

func ReluBackward(parent *Tensor) {
	child := parent.Children[0]
	if parent.IsScalar() {
		for i, _ := range child.Gradients {
			child.Gradients[i] += math.Max(0, parent.Backing.Backing[0]) * parent.Gradients[0]
		}
	} else {
		if !parent.TensorEqual(child) {
			panic("ReluBackward: Expected child tensor to be equal to parent")
		}

		for i, _ := range child.Gradients {
			child.Gradients[i] += math.Max(0, parent.Backing.Backing[i]) * parent.Gradients[i]
		}
	}
}

func FromTensorsBackward(parent *Tensor) {
	if len(parent.Children) == len(parent.Gradients) {
		for i := range parent.Children {
			child := parent.Children[i]
			if child.IsScalar() {
				child.Gradients[0] += parent.Gradients[i]
			} else {
				child.Gradients[0] += parent.AvgGradient()
			}
		}
		return
	}

	if len(parent.Children) == 0 {
		return
	}

	for i := range parent.Children {
		child := parent.Children[i]
		if len(parent.Gradients) > len(child.Gradients) {
			count := len(parent.Gradients) / len(child.Gradients)
			child.Gradients = parent.Gradients[i*count : i*count+count]
		} else {
			child.Gradients = Fill(child.Gradients, parent.AvgGradient())
		}
	}
}

func TanhBackward(parent *Tensor) {
	child := parent.Children[0]
	coshX := Cosh(parent.Backing)
	if parent.IsScalar() {
		for i, _ := range child.Gradients {
			child.Gradients[i] += (1 / (coshX.Backing[i] * coshX.Backing[i])) * parent.Gradients[0]
		}
	} else {
		if !parent.TensorEqual(child) {
			panic("ReluBackward: Expected child tensor to be equal to parent")
		}

		for i, _ := range child.Gradients {
			child.Gradients[i] += (1 / (coshX.Backing[i] * coshX.Backing[i])) * parent.Gradients[i]
		}
	}
}

func PowBackward(parent *Tensor) {
	first := parent.Children[0]
	second := parent.Children[1]
	x := first.Backing
	z := MapFloat64Slice(second.Backing.Backing, func(value float64, i int) float64 {
		return math.Pow(value, second.Backing.Backing[i])
	})

	if parent.IsScalar() {
		for i := range first.Gradients {
			first.Gradients[i] += second.Backing.Backing[i] * math.Pow(x.Backing[i], second.Backing.Backing[i]-1) * parent.Gradients[0]
			second.Gradients[i] += z[i] * math.Log(x.Backing[i]) * parent.Gradients[0]
		}
	} else {
		if !parent.TensorEqual(first) || !parent.TensorEqual(second) {
			panic("DivBackward: Expected child tensor to be equal to parent")
		}

		for i := range first.Gradients {
			first.Gradients[i] += second.Backing.Backing[i] * math.Pow(x.Backing[i], second.Backing.Backing[i]-1) * parent.Gradients[i]
			second.Gradients[i] += z[i] * math.Log(x.Backing[i]) * parent.Gradients[i]
		}
	}
}

func (t *Tensor) TensorEqual(other *Tensor) bool {
	return len(t.Backing.Backing) == len(other.Backing.Backing)
}

func (t *Tensor) ToString() string {
	return fmt.Sprintf("Tensor(data=%v, gradients=%v, children=%v)", t.Backing.ToString(), t.Gradients, len(t.Children))
}

func (t *Tensor) Flatten() *Tensor {
	t.Backing.Flatten()

	return t
}

func (t *Tensor) Reshape(dims ...int) *Tensor {
	t.Backing.Reshape(dims...)

	return t
}

func (t *Tensor) IsScalar() bool {
	return t.Backing.IsScalar()
}

func (t *Tensor) AvgGradient() float64 {
	sum := 0.0
	for _, value := range t.Gradients {
		sum += value
	}

	return sum / float64(len(t.Gradients))
}

func (t *Tensor) MaxValue() float64 {
	maxValue := float64(math.MinInt32)
	for _, value := range t.Backing.Backing {
		maxValue = math.Max(maxValue, value)
	}

	return maxValue
}

func (t *Tensor) Backward() {
	nodes := t.Topological()
	slices.Reverse(nodes)
	t.Gradients = Fill(t.Gradients, 1)

	for _, tensor := range nodes {
		if tensor.backward == nil {
			continue
		}

		tensor.backward(tensor)
	}
}

func (t *Tensor) Topological(tensors ...*Tensor) []*Tensor {
	if slices.Contains(tensors, t) {
		return tensors
	}

	for _, child := range t.Children {
		tensors = child.Topological(tensors...)
	}

	tensors = append(tensors, t)
	return tensors
}
func (t *Tensor) Zerograd() {
	for i := range t.Gradients {
		t.Gradients[i] = 0
	}
}

func (t *Tensor) Sum() *Tensor {
	sum := Sum(t.Backing)

	return &Tensor{
		sum,
		make([]float64, 1),
		[]*Tensor{t},
		SumBackward,
	}
}

func (t *Tensor) Log() *Tensor {
	log := Log(t.Backing)

	return &Tensor{
		log,
		make([]float64, len(log.Backing)),
		[]*Tensor{t},
		LogBackward,
	}
}

func (t *Tensor) Mse(expected *Tensor) (*Tensor, error) {
	if !t.TensorEqual(expected) {
		return nil, errors.New("expected tensor to be equal to tensor")
	}

	result, err := expected.Sub(t)

	if err != nil {
		return nil, err
	}

	result, err = TensorPow(result, 2.0)

	if err != nil {
		return nil, err
	}

	return result, nil
}

func (t *Tensor) CategoricalCrossEntropy(expected *Tensor) (*Tensor, error) {
	if !t.TensorEqual(expected) {
		return nil, errors.New("expected tensor to be equal to tensor")
	}

	logOfPrediction := t.Log()

	result, err := expected.Mul(logOfPrediction)

	if err != nil {
		return nil, err
	}

	return result.Sum().Negate()
}

func (t *Tensor) ExpandTo(length int) *Tensor {
	for i := len(t.Backing.Backing); i < length; i++ {
		t.Backing.Backing = append(t.Backing.Backing, t.Backing.Backing[i-1])
		t.Gradients = append(t.Gradients, t.Backing.Backing[i-1])
	}

	return t
}
func (t *Tensor) ensureOtherTensor(second interface{}) *Tensor {
	var otherTensor *Tensor
	switch otherValue := second.(type) {
	case *Tensor:
		otherTensor = otherValue
		if otherTensor.IsScalar() && !t.IsScalar() {
			otherTensor.ExpandTo(len(t.Backing.Backing))
		}

	case float64:
		otherTensor = NewTensorWithValue(len(t.Backing.Backing), otherValue)
	}

	if otherTensor == nil {
		panic("Expected a tensor op()")
	}

	return otherTensor
}

func op(first *Tensor, second interface{}, callback func(first *NArray, second *NArray) (*NArray, error), backward func(parent *Tensor)) (*Tensor, error) {
	otherTensor := first.ensureOtherTensor(second)
	result, err := callback(first.Backing, otherTensor.Backing)
	if err != nil {
		return nil, err
	}

	return &Tensor{
		result,
		make([]float64, len(result.Backing)),
		[]*Tensor{first, otherTensor},
		backward,
	}, nil
}

func TensorDiv[T *Tensor | float64](first *Tensor, other T) (*Tensor, error) {
	return op(first, other, Div, DivBackward)
}

func (t *Tensor) Div(other *Tensor) (*Tensor, error) {
	return TensorDiv(t, other)
}

func TensorAdd[T *Tensor | float64](first *Tensor, other T) (*Tensor, error) {
	return op(first, other, Add, AddBackward)
}

func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	return TensorAdd(t, other)
}

func TensorMul[T *Tensor | float64](first *Tensor, other T) (*Tensor, error) {
	return op(first, other, Mul, MulBackward)
}

func (t *Tensor) Mul(other *Tensor) (*Tensor, error) {
	return TensorMul(t, other)
}

func TensorPow[T *Tensor | float64](first *Tensor, other T) (*Tensor, error) {
	return op(first, other, Pow, PowBackward)
}

func (t *Tensor) Pow(other *Tensor) (*Tensor, error) {
	return TensorPow(t, other)
}

func (t *Tensor) Negate() (*Tensor, error) {
	return TensorMul(t, -1.0)
}

func TensorSub[T *Tensor | float64](first *Tensor, other T) (*Tensor, error) {
	otherTensor := first.ensureOtherTensor(other)
	otherTensor, err := otherTensor.Negate()
	if err != nil {
		return nil, err
	}

	return first.Add(otherTensor)
}

func (t *Tensor) Sub(other *Tensor) (*Tensor, error) {
	return TensorSub(t, other)
}

func (t *Tensor) Softmax() (*Tensor, error) {
	sum := t.Sum()
	result, err := t.Div(sum)

	if err != nil {
		return nil, err
	}

	return result, nil
}

func (t *Tensor) Relu() *Tensor {
	result := NewTensorEmpty(len(t.Backing.Backing))

	for i, _ := range t.Backing.Backing {
		result.Backing.Backing[i] = math.Max(0, t.Backing.Backing[i])
	}

	result.backward = ReluBackward
	result.Children = []*Tensor{t}
	return result
}

func (t *Tensor) Tanh() *Tensor {
	result := NewTensorEmpty(len(t.Backing.Backing))

	for i, _ := range t.Backing.Backing {
		result.Backing.Backing[i] = math.Tanh(t.Backing.Backing[i])
	}

	result.backward = TanhBackward
	result.Children = []*Tensor{t}
	return result
}
