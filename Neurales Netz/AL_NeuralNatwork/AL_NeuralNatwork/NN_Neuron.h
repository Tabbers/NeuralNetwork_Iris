#pragma once
#include <vector>
#include "NN_functions.h"
#include "NN_Types.h"

namespace nn
{
	class Neuron
	{
	public:
		Neuron();
		~Neuron();

		void CreateConnections(short);

		inline void EnableLearning() { m_learn = true; }
		inline void DisableLeanrning() { m_learn = false; }

		inline std::vector<float> GetInputWeights() { return m_inputWeights; };
		inline void SetInputWeights(std::vector<float> newWeights) { this->m_inputWeights = newWeights; };

		inline nn::functions::ActivationFunctions GetActivationfunction() { return m_functions; }
		inline void SetActivationfunction(nn::functions::ActivationFunctions func) { this->m_functions = func; }

		inline void SetOutput(float output) { this->m_previousOutput = output; }
		inline float GetOutput() { return m_previousOutput; }

		inline void SetInput(float input) { this->m_previousWeightedInput = input; }
		inline float Getinput() { return m_previousWeightedInput; }

		float DoWork(std::vector<float> &);
		std::vector<nn::types::CorrectionValueWeightPair> Learn(float, std::vector<float> &, std::vector<nn::types::CorrectionValueWeightPair>&, unsigned int , unsigned int);

		bool g_Output = true;
	private:
		bool m_learn = true;
		float m_previousOutput;
		float m_previousWeightedInput;
		std::vector<float> m_inputWeights;
		nn::functions::ActivationFunctions m_functions;
	};
}

