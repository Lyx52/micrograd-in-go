package tests

import (
	"fmt"

	"github.com/Lyx52/micrograd-in-go.git/nn"
)

func Test_MLP() {
	context := nn.NewNeuralContext(0)
	mlp := nn.NewModule(context,
		nn.NewLinearLayer(context, 3, 4, true, nn.TanhActivation),
		nn.NewLinearLayer(context, 4, 4, true, nn.TanhActivation),
		nn.NewLinearLayer(context, 4, 1, true, nn.NoneActivation),
	)

	xs := []*nn.Tensor{
		nn.NewTensor(2.0, 3.0, -1.0),
		nn.NewTensor(3.0, -1.0, 0.5),
		nn.NewTensor(0.5, 1.0, 1.0),
		nn.NewTensor(1.0, 1.0, -1.0),
	}

	ys := []*nn.Tensor{
		nn.NewTensor(1.0),
		nn.NewTensor(1.0),
		nn.NewTensor(-1.0),
		nn.NewTensor(1.0),
	}

	learningRate := 0.05
	steps := 1000

	for i := 0; i < steps; i++ {
		mlp.UpdateParameters(func(context *nn.NeuralContext, tensor *nn.Tensor) {
			for j := range tensor.Backing.Backing {
				tensor.Backing.Backing[j] += -learningRate * tensor.Gradients[j]
			}
		})

		mlp.Zerograd()

		loss, err := nn.CrossEntropyLoss(mlp, ys, xs, 10)
		if err != nil {
			panic(err)
		}
		loss.Backward()
		fmt.Println(fmt.Sprintf("[Step %d/%d] Loss: %f", i, steps, loss.Backing.Scalar()))
	}

	for i := range xs {
		result, err := mlp.Execute(xs[i])
		if err != nil {
			panic(err)
		}

		fmt.Println(fmt.Sprintf("%v = %v", xs[i].ToString(), result.ToString()))
	}
}
