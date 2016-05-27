#pragma once
#include <math.h>
namespace nn
{
	namespace functions
	{
		typedef float(*ActFunc)(float);

		struct ActivationFunctions
		{
			ActivationFunctions()
			{
			}
			ActivationFunctions(ActFunc forward, ActFunc backprop)
			{
				this->feedFrowardFunction = forward;
				this->backPorpagationFunction = backprop;
			};
			ActFunc feedFrowardFunction;
			ActFunc backPorpagationFunction;
		};

		//Sinusoid and the corresponding Derivated Function
		static float Sinusoid(float x)
		{
			return sinf(x);
		}
		static float SinusoidDerivation(float x)
		{
			return cosf(x);
		}

		//Tangent Hyperbolicus and the corresponding Derivated Function
		static float TanH(float x)
		{
			return tanh(x);
		}
		static float TanHDerivation(float x)
		{
			float intermediateValue = TanH(x);
			return 1 - (intermediateValue * intermediateValue);
		}

		//SoftSign and the corresponding Derivated Function
		static float SoftSign(float x)
		{
			return x/(1+ fabsf(x));
		}
		static float SoftSignDerivation(float x)
		{
			float intermediateValue = 1+fabsf(x);
			return 1 / (intermediateValue * intermediateValue);
		}

		//BinaryStep and the corresponding Derivated Function
		static float BinStep(float x)
		{
			if (x < 0)
			{
				return 0;
			}
			else
			{
				return 1;
			}
		}
		static float BinStepDerivation(float x)
		{
			if (x != 0)
			{
				return 0;
			}
		}

		//ReLU and the corresponding Derivated Function
		static float RelU(float x)
		{
			if (x < 0)
			{
				return 0;
			}
			else
			{
				return x;
			}
		}
		static float RelUDerivation(float x)
		{
			if (x < 0)
			{
				return 0;
			}
			else
			{
				return 1;
			}
		}

		//SoftMax
		static float Softmax(std::vector<float> input, unsigned int index)
		{
			float expCurrentOutput = expf(input[index]);
			float expSum = 0;
			for (unsigned int i = 0u; i < input.size(); ++i)
			{
				expSum += expf(input[i]);
			}
			return expCurrentOutput / expSum;
		}
	}
}