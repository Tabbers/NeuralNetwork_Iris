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
			netinput += data[i] * weights[i];
		}
		neuron.SetInput(netinput);
		float output = neuron.GetActivationfunction().feedFrowardFunction(netinput);
		neuron.SetOutput(output);
		return output;
	}

	static float CalculateDiscrapency(float desiredOutput, float actualOutput)
	{
		return desiredOutput - actualOutput;
	}

	static std::vector<nn::types::CorrectionValueWeightPair> NeuronLearn(nn::Neuron &neuron, float desiredOutput, std::vector<float> inputs, float netj, std::vector<nn::types::CorrectionValueWeightPair> &correctionValues, 
		unsigned int currenNeuronNumber, unsigned int numberOfNeurons)
	{
		std::vector<nn::types::CorrectionValueWeightPair> nextLayerPartialCorrectionVal;
		std::vector<float> weightsOld = neuron.GetInputWeights();
		std::vector<float> weightsNew;
		weightsNew.resize(weightsOld.size());
		float correctionValue = 0.0f;
		if (neuron.g_Output)
		{
			float previousOutput = neuron.GetOutput();
			for (unsigned int i = 0u; i < weightsNew.size(); ++i)
			{
				correctionValue = neuron.GetActivationfunction().backPorpagationFunction(netj)*CalculateDiscrapency(desiredOutput, previousOutput);
				float dWeight = (nn::types::LEARNRATE * correctionValue *  inputs[i]);
				weightsNew[i] = weightsOld[i] + dWeight;

				nn::types::CorrectionValueWeightPair pair;
				pair.EdgeCorrection = correctionValue;
				pair.EdgeWeight = weightsNew[i];

				nextLayerPartialCorrectionVal.push_back(pair);

			}
		}
		else
		{
			float summedOutgoingWeightsWithCorrectionValues = 0.0f;
			for (unsigned int j = 0u; j < weightsNew.size(); ++j)
			{
				unsigned int index = (currenNeuronNumber * (numberOfNeurons-1)) + j;
				summedOutgoingWeightsWithCorrectionValues += correctionValues[index].EdgeCorrection * correctionValues[index].EdgeWeight;
			}
			for (unsigned int i = 0u; i < weightsNew.size(); ++i)
			{
				correctionValue = neuron.GetActivationfunction().backPorpagationFunction(netj) * summedOutgoingWeightsWithCorrectionValues;
				float dWeight = (nn::types::LEARNRATE * correctionValue *  inputs[i]);
				weightsNew[i] = weightsOld[i] + dWeight;
				
				nn::types::CorrectionValueWeightPair pair;
				pair.EdgeCorrection = correctionValue;
				pair.EdgeWeight = weightsNew[i];

				nextLayerPartialCorrectionVal.push_back(pair);
			}
		}		
		neuron.SetInputWeights(weightsNew);
		return nextLayerPartialCorrectionVal;
	}

}

nn::Neuron::Neuron() : m_previousOutput(0.0f)
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
		float weight = Randomizer::GetRandom(-1.0f, 1.0f);
		m_inputWeights.push_back(weight);
	}
}

float nn::Neuron::DoWork(std::vector<float> &input)
{
	return NeuronDoWork(input, *this);
}

std::vector<nn::types::CorrectionValueWeightPair> nn::Neuron::Learn(float desiredOutput, std::vector<float> &inputs, std::vector<nn::types::CorrectionValueWeightPair> &prevLayersCorrectionValues, unsigned int NeuronNumber, unsigned int MaxNeurons)
{
	return NeuronLearn(*this, desiredOutput,inputs, this->m_previousWeightedInput, prevLayersCorrectionValues, NeuronNumber, MaxNeurons);
}
