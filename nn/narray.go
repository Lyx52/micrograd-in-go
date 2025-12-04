package nn

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

type NArray struct {
	Backing []float64
}

func NewNArray(values []float64) *NArray {
	return &NArray{
		Backing: values,
	}
}

func NewNArrayFromValues(values ...float64) *NArray {
	return NewNArray(values)
}

func NewNArrayFilledWith(length int, value float64) *NArray {
	otherNArrayValues := make([]float64, length)
	otherNArrayValues = Fill(otherNArrayValues, value)
	return NewNArray(otherNArrayValues)
}

func NewNArrayEmpty(length int) *NArray {
	return NewNArrayFilledWith(length, 0)
}

func NewNArrayRand(length int, rand *rand.Rand) *NArray {
	array := make([]float64, length)
	for i := range array {
		array[i] = rand.Float64()
	}

	return NewNArray(array)
}

func (t *NArray) ToString() string {
	return fmt.Sprintf("NArray(data=%v, len=%v)", t.Backing, len(t.Backing))
}

func (t *NArray) op(other interface{}, opCallback func(first float64, second float64) float64) (*NArray, error) {
	switch otherValue := other.(type) {
	case *NArray:
		result := make([]float64, len(t.Backing))
		if otherValue.IsScalar() {
			scalarValue := otherValue.Scalar()
			for i := range t.Backing {
				result[i] = opCallback(t.Backing[i], scalarValue)
			}
		} else {
			for i := range t.Backing {
				result[i] = opCallback(t.Backing[i], otherValue.Backing[i])
			}
		}

		return &NArray{
			Backing: result,
		}, nil
	case float64:
		result := make([]float64, len(t.Backing))
		otherNArrayValues := make([]float64, len(t.Backing))
		otherNArrayValues = Fill(otherNArrayValues, otherValue)
		otherNArray := NewNArray(otherNArrayValues)

		for i := range t.Backing {
			result[i] = opCallback(t.Backing[i], otherNArray.Backing[i])
		}

		return &NArray{
			Backing: result,
		}, nil
	}

	return nil, errors.New("unexpected other operation type")
}

func (t *NArray) Length() int {
	return len(t.Backing)
}

func (t *NArray) IsScalar() bool {
	return len(t.Backing) == 1
}

func (t *NArray) Scalar() float64 {
	if len(t.Backing) != 1 {
		panic("NArray is not a scalar value")
	}

	return t.Backing[0]
}

func Add[T float64 | *NArray](first *NArray, value T) (*NArray, error) {
	return first.op(value, func(first float64, second float64) float64 {
		return first + second
	})
}

func Sub[T float64 | *NArray](first *NArray, value T) (*NArray, error) {
	return first.op(value, func(first float64, second float64) float64 {
		return first - second
	})
}

func Div[T float64 | *NArray](first *NArray, value T) (*NArray, error) {
	return first.op(value, func(first float64, second float64) float64 {
		return first / second
	})
}

func Mul[T float64 | *NArray](first *NArray, value T) (*NArray, error) {
	return first.op(value, func(first float64, second float64) float64 {
		return first * second
	})
}

func Pow[T float64 | *NArray](first *NArray, value T) (*NArray, error) {
	return first.op(value, func(first float64, second float64) float64 {
		return math.Pow(first, second)
	})
}

func Negate(array *NArray) (*NArray, error) {
	return Mul(array, -1.0)
}

func Cosh(array *NArray) *NArray {
	result := NewNArrayEmpty(len(array.Backing))
	for i, value := range array.Backing {
		result.Backing[i] = math.Cosh(value)
	}

	return result
}

func Sum(array *NArray) *NArray {
	sum := 0.0
	for _, value := range array.Backing {
		sum += value
	}

	return &NArray{
		Backing: []float64{sum},
	}
}
