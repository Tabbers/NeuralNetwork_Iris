#pragma once
#include "NN_Neuron.h"
#include "NN_DataImport.h"
namespace nn
{
	class Layer
	{
	public:
		Layer();
		~Layer();

		void Init(short,short);
		std::vector<float> DoWork(std::vector<float> &);
		inline Neuron* GetNeurons() { return m_neurons; };
	private:
		short m_numberOfNeurons;
		Neuron* m_neurons;
	};
	class Network
	{
	public:
		Network(short, short, short);
		~Network();

		void Learn(std::vector<nn::types::PlantData> &);
	private:
		const float MIN_ACCAPTED_ERROR = 0.01f;
		float m_error;

		Layer* m_LayersHidden;
		Layer* m_Output;
	};


}
