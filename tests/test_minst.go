package tests

import (
	"fmt"
	"slices"
	"time"

	"github.com/Lyx52/micrograd-in-go.git/data"
	"github.com/Lyx52/micrograd-in-go.git/nn"
	"github.com/Lyx52/micrograd-in-go.git/types"
)

func Test_Minst(useRandom bool) {
	var err error
	_, labels, err := data.ReadIdx[uint8]("./public/train-labels.bin")
	if err != nil {
		panic(err)
	}

	dims, images, err := data.ReadIdx[uint8]("./public/train-images.bin")
	if err != nil {
		panic(err)
	}

	// Prep labels
	labelSet := data.NewSet[uint8]()
	labelSet.AddAll(labels)
	encoded := nn.OneHotEncode(slices.Collect[string](labelSet.Keys()))
	ys := types.MapSlice[*nn.Tensor, uint8](labels, func(value uint8, i int) *nn.Tensor {
		return nn.NewTensorFromArray(encoded[fmt.Sprint(value)])
	})

	// Prep images
	imageSize := dims[1] * dims[2]
	casted := types.CastNumber[float64, uint8](images, func(input uint8) float64 {
		return float64(input)
	})
	chunked := slices.Chunk[[]float64, float64](casted, imageSize)
	xs := types.MapIter[*nn.Tensor, []float64](chunked, func(value []float64, i int) *nn.Tensor {
		return nn.NewTensorFromArray(value).Reshape(dims[1:]...)
	})

	var context *nn.NeuralContext
	if useRandom {
		context = nn.NewNeuralContext(time.Now().UnixMilli())
	} else {
		context = nn.NewNeuralContext(0)
	}

	module := nn.NewModule(context,
		nn.NewFlattenLayer(context),
		nn.NewLinearLayer(context, 28*28, 128, true, nn.ReluActivation),
		nn.NewLinearLayer(context, 128, 64, true, nn.TanhActivation),
		nn.NewLinearLayer(context, 64, 32, true, nn.TanhActivation),
		nn.NewLinearLayer(context, 32, 10, true, nn.NoneActivation),
		nn.NewSoftmaxLayer(context),
	)

	learningRate := 0.05
	steps := 1000

	for i := 0; i < steps; i++ {
		module.UpdateParameters(func(context *nn.NeuralContext, tensor *nn.Tensor) {
			for j := range tensor.Backing.Backing {
				tensor.Backing.Backing[j] += -learningRate * tensor.Gradients[j]
			}
		})

		module.Zerograd()

		loss, err := nn.CrossEntropyLoss(module, ys, xs, 10)
		if err != nil {
			panic(err)
		}
		loss.Backward()
		fmt.Println(fmt.Sprintf("[Step %d/%d] Loss: %f", i, steps, loss.Backing.Scalar()))
	}

	//for i := range xs {
	//	result, err := module.Execute(xs[i])
	//	if err != nil {
	//		panic(err)
	//	}
	//
	//	fmt.Println(fmt.Sprintf("%v = %v", xs[i].ToString(), result.ToString()))
	//}
}
