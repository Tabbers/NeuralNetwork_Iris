#include "NN_Structure.h"

namespace
{
	static std::vector<float> SoftMaxOutput(std::vector<float> &input)
	{
		std::vector<float> result;
		result.reserve(input.size());
		for (unsigned int i = 0u; i < input.size(); ++i)
		{
			float res = nn::functions::Softmax(input, i);
			result.push_back(res);
		}
		return result;
	}

	static float CalculateError(std::vector<float>& teachingValues, std::vector<float>& outputs)
	{
		for (unsigned int i = 0u; i < teachingValues.size(); ++i)
		{
			if (teachingValues[i] == 1)
			{
				return fabsf(teachingValues[i] - outputs[i]);
			}
		}
	}
}


nn::Layer::Layer()
{	
}


nn::Layer::~Layer()
{
	delete[] m_neurons;
}

void nn::Layer::Init(short numberOfNeurons, short numOfInputs)
{
	m_numberOfNeurons = numberOfNeurons;
	m_neurons = new Neuron[m_numberOfNeurons];
	for (unsigned int i = 0u; i < m_numberOfNeurons; ++i)
	{
		m_neurons[i].CreateConnections(numOfInputs);
	}
}

std::vector<float> nn::Layer::DoWork(std::vector<float> &Layerinput)
{
	std::vector<float> output;
	output.reserve(m_numberOfNeurons);
	for (unsigned int i = 0u; i < m_numberOfNeurons; ++i)
	{
		float outputSingle = m_neurons[i].DoWork(Layerinput);
		output.push_back(outputSingle);
	}
	return output;
}

void nn::Layer::Learn(std::vector<float>& teachingValue,std::vector<float> &inputs, std::vector<nn::types::CorrectionValueWeightPair>& prevLayersCorrectionValues, unsigned int numberOfNeurons)
{
	std::vector<nn::types::CorrectionValueWeightPair> nextLayersCorrectionValues;
	for (unsigned int i = 0u; i < m_numberOfNeurons; ++i)
	{
		std::vector<nn::types::CorrectionValueWeightPair> temp =  m_neurons[i].Learn(teachingValue[i], inputs, prevLayersCorrectionValues, i, numberOfNeurons);
		nextLayersCorrectionValues.reserve(nextLayersCorrectionValues.size() + temp.size());
		for (unsigned int j = 0u; j < temp.size(); ++j)
		{
			nextLayersCorrectionValues.push_back(temp[j]);
		}

	}
	prevLayersCorrectionValues = nextLayersCorrectionValues;
}


//====================================================================================
// NETWORK
//====================================================================================

nn::Network::Network(short numOfHiddenLayers, short numOfInputs, short numOfOutputs) :m_error(1.0f)
{
	m_NumOfHiddenLayers = numOfHiddenLayers;
	m_LayersHidden = new Layer[m_NumOfHiddenLayers];
	for (unsigned int i = 0u; i < m_NumOfHiddenLayers; ++i)
	{
		m_LayersHidden[i].Init(nn::NUMBER_OF_NEURONS, numOfInputs);
	}
	m_Output = new Layer();
	m_Output->Init(numOfOutputs,nn::NUMBER_OF_NEURONS);
}

nn::Network::~Network()
{
	delete[] m_LayersHidden;
}

void nn::Network::Learn(std::vector<nn::types::PlantData>&learningDatabase)
{
	unsigned int i = 0;
	while (true)
	{
		std::vector<nn::types::PlantData> data = nn::fileimport::ShuffleData(learningDatabase);
		for (unsigned int i = 0u; i < data.size(); ++i)
		{
			//Input the Data into the Hidden Layer
			std::vector<float> teachingValues;
			std::vector<float> Entry = nn::fileimport::GetDataEntry(i, data, teachingValues);
			std::vector<float> HiddenOutput;
			std::vector<float> OutputOutput;
			std::vector<float> SoftmaxOutput;
			std::vector<nn::types::CorrectionValueWeightPair> correctionData;
			//Input Hidden Layer Results into Output
			for (unsigned int i = 0u; i < Entry.size(); ++i)
			{
				Entry[i] = nn::types::normalizeInputs(Entry[i], 0.0f, 8.0f, -1.0f, 1.0f);
			}
			HiddenOutput = m_LayersHidden->DoWork(Entry);
			OutputOutput = m_Output->DoWork(HiddenOutput);
			SoftmaxOutput = SoftMaxOutput(OutputOutput);
			for (unsigned int j = 0u; j < SoftmaxOutput.size(); ++j)
			{
				Neuron* output = m_Output->GetNeurons();
				output[j].SetOutput(SoftmaxOutput[j]);
			}
			
			m_error = CalculateError(teachingValues, SoftmaxOutput);
			if (m_error < MIN_ACCAPTED_ERROR)
			{
				std::cout << "Error: " << m_error << " Learningcycle: " << i << std::endl;
				break;
			}
			else
			{
				std::cout << "Error: " << m_error << " Learningcycle: " << i << std::endl;
			}
			//Backpropagete the Output layer
			m_Output->Learn(teachingValues, HiddenOutput,correctionData, 0);
			//Needs to be pushed in for the Learnfunctions in later Layers to work => number of Techingvalues must match Neuron Number
			teachingValues.push_back(0.0f);
			// backrpopagate the Hidden Layer
			m_LayersHidden->Learn(teachingValues, Entry, correctionData, m_Output->GetNumberOfNeurons());
		}
		++i;
	}
}


