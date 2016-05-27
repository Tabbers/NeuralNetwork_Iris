
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
	functions.backPorpagationFunction = nn::functions::TanHDerivation;
	HiddenNeurons[0].SetActivationfunction(functions);
	HiddenNeurons[0].g_Output = false;

	functions.feedFrowardFunction = nn::functions::Sinusoid;
	functions.backPorpagationFunction = nn::functions::SinusoidDerivation;
	HiddenNeurons[1].SetActivationfunction(functions);
	HiddenNeurons[1].g_Output = false;

	functions.feedFrowardFunction = nn::functions::RelU;
	functions.backPorpagationFunction = nn::functions::RelUDerivation;
	HiddenNeurons[2].SetActivationfunction(functions);
	HiddenNeurons[2].g_Output = false;

	functions.feedFrowardFunction = nn::functions::SoftSign;
	functions.backPorpagationFunction = nn::functions::SoftSignDerivation;
	HiddenNeurons[3].SetActivationfunction(functions);
	HiddenNeurons[3].g_Output = false;

	//Set the feed Forward ans backpropagation Functions of the Output layer
	functions.feedFrowardFunction = nn::functions::TanH;
	functions.backPorpagationFunction = nn::functions::TanHDerivation;
	OutputNeurons[0].SetActivationfunction(functions);
	OutputNeurons[0].g_Output = true;

	functions.feedFrowardFunction = nn::functions::TanH;
	functions.backPorpagationFunction = nn::functions::TanHDerivation;
	OutputNeurons[1].SetActivationfunction(functions);
	OutputNeurons[1].g_Output = true;

	functions.feedFrowardFunction = nn::functions::TanH;
	functions.backPorpagationFunction = nn::functions::TanHDerivation;
	OutputNeurons[2].SetActivationfunction(functions);
	OutputNeurons[2].g_Output = true;
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
