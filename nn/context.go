package nn

import (
	rand2 "math/rand"
)

type NeuralContext struct {
	Random *rand2.Rand
}

func NewNeuralContext(seed int64) *NeuralContext {
	source := rand2.NewSource(seed)
	return &NeuralContext{
		Random: rand2.New(source),
	}
}
