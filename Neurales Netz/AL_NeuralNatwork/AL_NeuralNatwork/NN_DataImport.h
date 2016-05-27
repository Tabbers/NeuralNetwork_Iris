#pragma once
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "NN_Types.h"

namespace nn
{
	const short NUMBER_OF_NEURONS = 4;
	namespace fileimport
	{
		static char* GetSubstring(char* start, char* end)
		{
			int distance = end - start;
			char* substr = new char[distance + 1];
			strncpy(substr, start, distance);
			substr[distance] = '\0';
			return substr;
		}

		static std::vector<float> GetDataEntry(unsigned int index, std::vector<types::PlantData> &data, std::vector<float> &teachingValues)
		{
			std::vector<float> singleEntry;
			singleEntry.reserve(data[index].numberOfEntries);
			for (unsigned int i = 0u; i < data[index].numberOfEntries; i++)
			{
				singleEntry.push_back(data[index].Array[i]);
				
			}
			if (data[index].plantClass == "Iris-setosa")
			{
				teachingValues.push_back(1.0f);
				teachingValues.push_back(0.0f);
				teachingValues.push_back(0.0f);
			}
			else if (data[index].plantClass == "Iris-versicolor")
			{
				teachingValues.push_back(0.0f);
				teachingValues.push_back(1.0f);
				teachingValues.push_back(0.0f);
			}
			else if (data[index].plantClass == "Iris-virginica")
			{
				teachingValues.push_back(0.0f);
				teachingValues.push_back(0.0f);
				teachingValues.push_back(1.0f);
			}
			return singleEntry;
		}

		static 	std::vector<types::PlantData> ShuffleData(std::vector<types::PlantData> &data)
		{
			std::vector<types::PlantData> datacopy = data;
			std::random_shuffle(datacopy.begin(), datacopy.end());
			return datacopy;
		}

		static void SplitIntoLeaningAndTestData(std::vector<types::PlantData> &data, std::vector<types::PlantData> &learningdata, std::vector<types::PlantData> &testdata)
		{
			//Split data by 3s
			// user 80% of the data as learning data
			float learningDataAmount = data.size()*0.8;
			float testingDataAmount = data.size()*0.2;

			learningdata.reserve(learningDataAmount);
			testdata.reserve(testingDataAmount);

			for (unsigned int i = 0u; i < 40; ++i)
			{
				learningdata.push_back(data[i]);
			}
			for (unsigned int i = 40u; i < 50; ++i)
			{
				testdata.push_back(data[i]);
			}

			for (unsigned int i = 50; i < 90; ++i)
			{
				learningdata.push_back(data[i]);
			}
			for (unsigned int i = 90u; i < 100; ++i)
			{
				testdata.push_back(data[i]);
			}

			for (unsigned int i = 100; i < 140; ++i)
			{
				learningdata.push_back(data[i]);
			}
			for (unsigned int i = 140u; i < 150; ++i)
			{
				testdata.push_back(data[i]);
			}
		}
		static bool ReadData(char* fileName, std::vector<nn::types::PlantData> &data)
		{
			std::ifstream file;
			file.open(fileName);
			if (!file.is_open())
			{
				std::cout << "File " << fileName << " could not be opened" << std::endl;
				return false;
			}
			char* filecontent;
			size_t fileLength;
			// Find end of file to get file size
			file.seekg(0, file.end);
			fileLength = file.tellg();
			file.seekg(0, file.beg);

			// Allocate and read
			char* buffer = new char[fileLength + 1u];
			file.read(buffer, fileLength);
			buffer[fileLength] = '\0';
			filecontent = buffer;
			file.close();

			unsigned int i = 0;
			unsigned int DataIndex = 0;
			char* startLine = filecontent;
			char* endLine = startLine;

			types::PlantData datum;

			while (i < fileLength)
			{
				if (*endLine == ',')
				{
					char* substr = GetSubstring(startLine, endLine);
					datum.Array[DataIndex] = atof(substr);
					DataIndex++;
					startLine = endLine + 1;
				}
				if (*endLine == '\n')
				{
					DataIndex = 0;
					char* substr = GetSubstring(startLine, endLine);
					datum.plantClass = substr;
					data.push_back(datum);
					startLine = endLine + 1;
				}
				endLine++;
				++i;
			}
			if (buffer != nullptr)
			{
				delete buffer;
			}
		}
	}
}