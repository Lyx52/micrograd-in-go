package data

import (
	"fmt"
	"iter"
	"maps"
)

type Set[TValue any] struct {
	backing map[string]TValue
}

func NewSet[TValue any]() *Set[TValue] {
	return &Set[TValue]{make(map[string]TValue)}
}

func (set *Set[TValue]) AddAll(values []TValue) {
	for i := range values {
		set.Add(values[i])
	}
}

func (set *Set[TValue]) Add(value TValue) {
	set.backing[fmt.Sprint(value)] = value
}

func (set *Set[TValue]) Remove(value TValue) {
	delete(set.backing, fmt.Sprint(value))
}

func (set *Set[TValue]) Contains(value TValue) bool {
	_, ok := set.backing[fmt.Sprint(value)]
	return ok
}

func (set *Set[TValue]) Len() int {
	return len(set.backing)
}

func (set *Set[TValue]) Clear() {
	set.backing = make(map[string]TValue)
}

func (set *Set[TValue]) Values() iter.Seq[TValue] {
	return maps.Values(set.backing)
}

func (set *Set[TValue]) Keys() iter.Seq[string] {
	return maps.Keys(set.backing)
}
