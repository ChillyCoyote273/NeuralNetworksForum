from Perceptron import Perceptron


def activation(x):
	return 1 if x >= 0 else 0


def main():
	inputs = [
		[1.25, 1.25],
		[1, 1],
		[0.5, 0],
		[0.25, 1],
		[0.1, 0.5],
		[0.5, 1.5]
	]
	outputs = [0, 0, 0, 1, 1, 1]

	p = Perceptron(2)

	for i in range(100):
		p.learn(inputs, outputs, 0.05)
	
	for inp, correct in zip(inputs, outputs):
		output = activation(p.forward(inp))
		print(f"Expected: {correct} Got: {output}")


if __name__ == "__main__":
	main()
