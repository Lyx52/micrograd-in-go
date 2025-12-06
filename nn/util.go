package nn

import (
	"math/rand"
	"sort"

	"github.com/Lyx52/micrograd-in-go.git/types"
)

func Fill[T any](arr []T, value T) []T {
	for i := 0; i < len(arr); i++ {
		arr[i] = value
	}

	return arr
}

func GetTotalElements[TInput int | int32](dims []TInput) int {
	if len(dims) == 0 {
		return 0
	}

	total := dims[0]

	for i := 0; i < len(dims)-1; i++ {
		total *= dims[i+1]
	}

	return int(total)
}

func MapFloat64Slice(arr []float64, mapCallback func(float64, int) float64) []float64 {
	result := make([]float64, len(arr))
	for i := 0; i < len(arr); i++ {
		result[i] = mapCallback(arr[i], i)
	}

	return result
}

func sampleIndices(arr []*Tensor, count int) []int {
	indices := make([]int, len(arr))
	for i := range arr {
		indices[i] = i
	}

	if len(indices) < count {
		return indices
	}

	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	return indices[:count]
}

func SampleDataset(ys []*Tensor, xs []*Tensor, batchSize int) ([]*Tensor, []*Tensor) {
	indices := sampleIndices(xs, batchSize)
	samplesX, samplesY := make([]*Tensor, len(indices)), make([]*Tensor, len(indices))
	for i := 0; i < len(indices); i++ {
		samplesX[i] = xs[indices[i]]
		samplesY[i] = ys[indices[i]]
	}

	return samplesX, samplesY
}

func OneHotEncodeAny(labels []any) map[string][]float64 {
	return OneHotEncode(types.ToStringSlice(labels))
}

func OneHotEncode(labels []string) map[string][]float64 {
	sort.Strings(labels)
	mapping := make(map[string][]float64)
	step := 0

	for i := 0; i < len(labels); i++ {
		mapping[labels[i]] = make([]float64, len(labels))
		mapping[labels[i]][step] = 1.0
		step++
	}

	return mapping
}
