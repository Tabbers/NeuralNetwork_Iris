
#include "NN_Types.h"
#include "NN_DataImport.h"
#include "NN_Structure.h"

void main()
{
	std::vector<nn::types::PlantData> plantDatabase;
	std::vector<nn::types::PlantData> learningDatabase;
	std::vector<nn::types::PlantData> testingDatabase;
	plantDatabase.reserve(150);
	nn::fileimport::ReadData("../iris/iris_original.data", plantDatabase);
	nn::fileimport::SplitIntoLeaningAndTestData(plantDatabase, learningDatabase, testingDatabase);
	
	const short hiddenlayers = 1;
	const short numberOfInputValues = 4;
	const short numberOfOutputs = 3;
	nn::Network* network = new nn::Network(hiddenlayers, numberOfInputValues, numberOfOutputs);

	network->Learn(learningDatabase);
	
	getchar();

	delete network;
}