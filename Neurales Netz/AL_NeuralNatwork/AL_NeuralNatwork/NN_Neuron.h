#pragma once
#include <vector>
#include "NN_functions.h"

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

		inline float GetBias() { return m_bias; };
		inline void  SetBias(float newBias) { this->m_bias = newBias; };

		inline nn::functions::ActivationFunctions GetActivationfunction() { return m_functions; }
		inline void SetActivationfunction(nn::functions::ActivationFunctions func) { this->m_functions = func; }

		inline void SetOutput(float output) { this->m_previousOutput = output; }
		inline float GetOutput() { return m_previousOutput; }

		float DoWork(std::vector<float> &);
		void Learn();

	private:
		bool m_learn = true;
		float m_previousOutput;
		float m_bias;
		std::vector<float> m_inputWeights;
		nn::functions::ActivationFunctions m_functions;
	};
}

