#include "NN_Structure.h"



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
	m_neurons->CreateConnections(numOfInputs);
}
std::vector<float> nn::Layer::DoWork(std::vector<float> &Layerinput)
{
	std::vector<float> output;
	output.reserve(m_numberOfNeurons);
	for (unsigned int i = 0u; i < m_numberOfNeurons; i)
	{
		output[i] = m_neurons[i].DoWork(Layerinput);
	}
}

nn::Network::Network(short numOfHiddenLayers, short numOfInputs, short numOfOutputs)
{
	m_LayersHidden = new Layer[numOfHiddenLayers];
	for (unsigned int i = 0u; i < numOfHiddenLayers; ++i)
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

	while (m_error > MIN_ACCAPTED_ERROR)
	{
		m_LayersHidden->DoWork();
	}
}
