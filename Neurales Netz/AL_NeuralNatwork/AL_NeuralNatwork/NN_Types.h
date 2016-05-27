#pragma once
#include <string>
namespace nn
{
	namespace types
	{
		const float LEARNRATE = 0.10f;
		struct PlantData
		{
			union
			{
				struct
				{
					float sepalLength;
					float sepalWidth;
					float petalLength;
					float petalWidth;
				};
				float Array[4];
			};
			std::string plantClass;
			unsigned int numberOfEntries = 4;
		};
		struct CorrectionValueWeightPair
		{
			float EdgeCorrection;
			float EdgeWeight;
		};

		static float normalizeInputs(float x, float dl, float dh, float nl, float nh )
		{
			return ((x - dl) *(nh - nl)) / (dh - dl) + nl;
		}
		static float denormalizeInputs(float x, float dl, float dh, float nl, float nh)
		{
			return ((dl - dh) * x -(nh * dl) +(dh * nl)) / (dl - dh);
		}
	}
}