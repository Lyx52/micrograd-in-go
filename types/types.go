package types

import (
	"fmt"
	"iter"
	"slices"
)

func ToStringSlice(arr []any) []string {
	result := make([]string, len(arr))
	for i := 0; i < len(arr); i++ {
		result[i] = fmt.Sprint(arr[i])
	}

	return result
}

func BoxValues[T GenericNumber](arr []T) []any {
	result := make([]any, len(arr))

	for i := 0; i < len(arr); i++ {
		result[i] = arr[i]
	}

	return result
}

func MapSlice[TResult any, TInput any](arr []TInput, mapCallback func(TInput, int) TResult) []TResult {
	result := make([]TResult, len(arr))
	for i := 0; i < len(arr); i++ {
		result[i] = mapCallback(arr[i], i)
	}

	return result
}

func MapIter[TResult any, TInput any](seq iter.Seq[TInput], mapCallback func(TInput, int) TResult) []TResult {
	arr := slices.Collect(seq)
	return MapSlice(arr, mapCallback)
}

type GenericNumber = interface {
	float64 | float32 | int64 | int32 | int16 | int8 | uint64 | uint32 | uint16 | uint8
}

func CastNumber[TResult GenericNumber, TInput GenericNumber](arr []TInput, castCallback func(TInput) TResult) []TResult {
	result := make([]TResult, len(arr))
	for i := 0; i < len(arr); i++ {
		result[i] = castCallback(arr[i])
	}

	return result
}
