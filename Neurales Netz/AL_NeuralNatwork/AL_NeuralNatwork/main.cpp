
#include "NN_Types.h"
#include "NN_DataImport.h"
#include "NN_Structure.h"
#include "NN_Neuron.h"

void Init(nn::Network* net)
{
	nn::Neuron* HiddenNeurons = net->GetHiddenLayer()->GetNeurons();
	nn::Neuron* OutputNeurons = net->GetOutputLayer()->GetNeurons();
	nn::functions::ActivationFunctions functions;

	//Set the feed forward and Backpropagation Functions of the Hidden Layer
	functions.feedFrowardFunction = nn::functions::TanH;
	functions.feedFrowardFunction = nn::functions::TanHDerivation;
	HiddenNeurons[0].SetActivationfunction(functions);

	functions.feedFrowardFunction = nn::functions::Sinusoid;
	functions.feedFrowardFunction = nn::functions::SinusoidDerivation;
	HiddenNeurons[1].SetActivationfunction(functions);

	functions.feedFrowardFunction = nn::functions::RelU;
	functions.feedFrowardFunction = nn::functions::RelUDerivation;
	HiddenNeurons[2].SetActivationfunction(functions);

	functions.feedFrowardFunction = nn::functions::SoftSign;
	functions.feedFrowardFunction = nn::functions::SoftSignDerivation;
	HiddenNeurons[3].SetActivationfunction(functions);

	//Set the feed Forward ans backpropagation Functions of the Output layer
	functions.feedFrowardFunction = nn::functions::BinStep;
	functions.feedFrowardFunction = nn::functions::BinStepDerivation;
	OutputNeurons[0].SetActivationfunction(functions);

	functions.feedFrowardFunction = nn::functions::BinStep;
	functions.feedFrowardFunction = nn::functions::BinStepDerivation;
	OutputNeurons[1].SetActivationfunction(functions);

	functions.feedFrowardFunction = nn::functions::BinStep;
	functions.feedFrowardFunction = nn::functions::BinStepDerivation;
	OutputNeurons[2].SetActivationfunction(functions);

}

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

	Init(network);

	network->Learn(learningDatabase);

	getchar();

	delete network;
}
