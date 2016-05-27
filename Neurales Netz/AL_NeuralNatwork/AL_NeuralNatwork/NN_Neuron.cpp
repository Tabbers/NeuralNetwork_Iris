#include "NN_Neuron.h"
#include "NN_Types.h"
#include "Randomizer.h"
namespace
{
	static float NeuronDoWork(std::vector<float> &data, nn::Neuron &neuron)
	{
		// input mit weight multiplizieren
		// sum all inputs + bias == netinput
		// insert into activation function 0> output
		float netinput = 0;
		std::vector<float> weights = neuron.GetInputWeights();
		for (unsigned int i = 0u; i < weights.size(); ++i)
		{
			data[i] = nn::types::normalizeInputs(data[i], 0.0f, 8.0f, 0.0f, 1.0f);
		}
		for (unsigned int i = 0u; i < weights.size(); ++i)
		{
			netinput += data[i] * weights[i];
		}
		netinput += neuron.GetBias();

		float output = neuron.GetActivationfunction().feedFrowardFunction(netinput);
		neuron.SetOutput(output);
		return output;
	}

	static void NeuronLearn(nn::Neuron &neuron)
	{
		float previousInput = neuron.GetOutput();
		std::vector<float> weights = neuron.GetInputWeights();
		for (unsigned int i = 0u; i < weights.size(); ++i)
		{
			weights[i] = neuron.GetActivationfunction().backPorpagationFunction(previousInput)*();
		}
		neuron.SetInputWeights(weights);
	}

}

nn::Neuron::Neuron() : m_bias(Randomizer::GetRandom(-0.5f, 0.5f))
					 , m_previousOutput(0.0f)
{
}


nn::Neuron::~Neuron()
{
}

void nn::Neuron::CreateConnections(short numberOfINputs)
{
	m_inputWeights.reserve(numberOfINputs);
	for (unsigned int i = 0u; i < numberOfINputs; i++)
	{
		float weight = 1.0f;
		m_inputWeights.push_back(weight);
	}
}

float nn::Neuron::DoWork(std::vector<float> &input)
{
	return NeuronDoWork(input, *this);
}

void nn::Neuron::Learn()
{
}
