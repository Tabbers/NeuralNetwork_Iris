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
	for (unsigned int i = 0u; i < m_numberOfNeurons; ++i)
	{
		float outputSingle = m_neurons[i].DoWork(Layerinput);
		output.push_back(outputSingle);
	}
	return output;
}


//====================================================================================
// NETWORK
//====================================================================================

nn::Network::Network(short numOfHiddenLayers, short numOfInputs, short numOfOutputs) :m_error(1)
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

	while (m_error > MIN_ACCAPTED_ERROR)
	{
		std::vector<nn::types::PlantData> data = nn::fileimport::ShuffleData(learningDatabase);
		for (unsigned int i = 0u; i < data.size(); ++i)
		{
			//Input the Data into the Hidden Layer
			std::vector<float> Entry = nn::fileimport::GetDataEntry(i, data);
			Entry = m_LayersHidden->DoWork(Entry);
		
			//Input Hidden Layer Results into Output
			Entry = m_Output->DoWork(Entry);

		}
	}
}
