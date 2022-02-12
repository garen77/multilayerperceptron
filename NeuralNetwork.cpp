//
//  NeuralNetwork.cpp
//  NeuralNetwork
//
//  Created by Crescenzo Garofalo on 01/02/22.
//

#include "NeuralNetwork.hpp"

bool isLogActive = false;
const char* LAYER = "Layer";
const char* NEURON = "Neuron";

double sigmoid(double inp) {
    return 1/(1 + std::exp(-inp));
}
NeuralNetwork::NeuralNetwork(int* conf, int nl) :configuration(conf), numOfLayers(nl) {
    
    /*
     nl : num layers
     [niumInput, numHiddenNeuron,..,numHiddenNeuron, numOutNeuron] -> [nl + 1]
    */
    
    this->layers = new Layer*[nl];
    
    for (int l = 1; l < nl + 1; l++) {
        int numInputs = this->configuration[l-1];
        int numOvInputsPlusBias = numInputs + 1;
        int numOutputs = this->configuration[l];
        Layer* layer = new Layer();
        layer->neurons = new Neuron*[numOutputs];
        layer->numOfNeurons = numOutputs;
        for (int i=0; i<numOutputs; i++) {
            Neuron* neuron = new Neuron(0.0, new double[numOvInputsPlusBias],0.0,numInputs);
            for (int j=0; j<numInputs; j++) {
                neuron->weights[j] = (((double) rand()) / (double) RAND_MAX) * (1 + 1) - 1;
            }
            neuron->weights[numInputs] = (((double) rand()) / (double) RAND_MAX) * (1 + 1) - 1;
            layer->neurons[i] = neuron;
        }
        this->layers[l-1] = layer;
    }
}

NeuralNetwork::NeuralNetwork(const char* weightsFileName) {
    
    FILE* weightsFile = fopen(weightsFileName, "r");
    
    char line[MAX_LINE_LENGTH];
    
    if(weightsFile == 0) {
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<std::vector<double>*>*>* layers = new std::vector<std::vector<std::vector<double>*>*>();
    std::vector<std::vector<double>*>* currLayer = nullptr;
    std::vector<double>* currNeuron = nullptr;
    while(fgets(line, MAX_LINE_LENGTH, weightsFile)) {
        if (strncmp(line, LAYER, strlen(LAYER)) == 0) {
            currLayer = nullptr;
            continue;
        }
        if (strncmp(line, NEURON, strlen(NEURON)) == 0) {
            currNeuron = nullptr;
            if(currLayer == nullptr) {
                currLayer = new std::vector<std::vector<double>*>();
                layers->push_back(currLayer);
            }
            continue;
        }
        if(currNeuron == nullptr) {
            currNeuron = new std::vector<double>();
            currLayer->push_back(currNeuron);
        }
        currNeuron->push_back(atof(line));
    }
    
    // init network
    int numLayers = (int)layers->size();
    this->configuration = new int[numLayers+1];
    for (int l = 1; l <= numLayers; l++) {
        if(l==1) {
            this->configuration[l-1] = (int)layers->at(l-1)->at(0)->size()-1;
        }
        this->configuration[l] = (int)layers->at(l-1)->size();
    }
    
    this->layers = new Layer*[numLayers];
    this->numOfLayers = numLayers;
    
    for (int l = 1; l < numLayers + 1; l++) {
        int numInputs = this->configuration[l-1];
        int numOvInputsPlusBias = numInputs + 1;
        int numOutputs = this->configuration[l];
        Layer* layer = new Layer();
        std::vector<std::vector<double>*>* layerFromFile = layers->at(l-1);
        layer->neurons = new Neuron*[numOutputs];
        layer->numOfNeurons = numOutputs;
        for (int i=0; i<numOutputs; i++) {
            Neuron* neuron = new Neuron(0.0, new double[numOvInputsPlusBias],0.0,numInputs);
            std::vector<double>* neuronFromFile = layerFromFile->at(i);
            for (int j=0; j<numInputs; j++) {
                neuron->weights[j] = neuronFromFile->at(j);
            }
            neuron->weights[numInputs] = neuronFromFile->at(numInputs);
            layer->neurons[i] = neuron;
        }
        this->layers[l-1] = layer;
    }

    // delete auxiliary vectors
    for(int i=0; i<numLayers; i++) {
        std::vector<std::vector<double>*>* layer = layers->at(i);
        for(int j=0; j<layer->size();j++) {
            std::vector<double>* neuron = layer->at(j);
            delete neuron;
        }
        delete layer;
    }
    delete layers;
}

NeuralNetwork::~NeuralNetwork() {
    for(int i=0; i<this->numOfLayers; i++) {
        delete this->layers[i];
    }
    delete[] this->layers;
}

double NeuralNetwork::activate(Neuron* neuron, double* inputs) {
    double sum = 0.0;
    int dim = neuron->numOfInputs;
    double* weights = neuron->weights;
    for (int i=0; i<dim; i++) {
        sum += weights[i]*inputs[i];
    }
    sum += weights[dim]; // bias sum
    return sigmoid(sum);
}

double* NeuralNetwork::frowardPropagate(double* inputs, int dim) {
    if(isLogActive) {
        cout<<"\nForwardPropagate\n";
    }
    double* currentInputs = new double[dim];
    for(int i=0; i<dim; i++) {
        currentInputs[i] = inputs[i];
    }
    for(int l=0; l<this->numOfLayers; l++) {
        if(isLogActive) {
            cout<<"\nlayer "<<l;
        }
        Layer* layer = this->layers[l];
        int layerSize = layer->numOfNeurons;
        double* newInputs = new double[layerSize];
        for(int n=0; n<layerSize; n++) {
            Neuron* neuron = layer->neurons[n];
            double neuronOut = this->activate(neuron, currentInputs);
            neuron->output = neuronOut;
            newInputs[n] = neuronOut;
            if(isLogActive) {
                cout<<"\nneuronOut="<<neuronOut;
            }
        }
        delete[] currentInputs;
        currentInputs = newInputs;
    }
    
    return currentInputs;
}

void NeuralNetwork::backPropagate(double* expected, int dim) {
    if(isLogActive) {
        cout<<"\nBackPropagate\n";
    }
    for(int i=this->numOfLayers-1; i>=0; i--) {
        if(isLogActive) {
            cout<<"\nlayer "<<i;
        }
        Layer* layer = this->layers[i];
        int layerSize = layer->numOfNeurons;
        double* errors = new double[layerSize];
        if(i != this->numOfLayers-1) {
            for(int j=0; j<layerSize; j++) {
                double error = 0.0;
                Layer* nextLayer = this->layers[i+1];
                int nextLayerSize = nextLayer->numOfNeurons;
                for(int n=0; n<nextLayerSize; n++) {
                    Neuron* neuron = nextLayer->neurons[n];
                    double* w = neuron->weights;
                    double delta = neuron->delta;
                    error += w[j]*delta;
                }
                errors[j] = error;
            }
        } else {
            for(int j=0; j<layerSize; j++) {
                Neuron* neuron = layer->neurons[j];
                double d = expected[j];
                double neuronOut = neuron->output;
                errors[j] = d - neuronOut;
                if(isLogActive) {
                    cout<<"\nexpected["<<j<<"]="<<d<<" out from neuron="<<neuronOut;
                }
            }
        }
        for(int j=0; j<layerSize; j++) {
            Neuron* neuron = layer->neurons[j];
            double neuronOut = neuron->output;
            double neuronDelta = errors[j]*neuronOut*(1-neuronOut);
            neuron->delta = neuronDelta;
            if(isLogActive) {
                cout<<"\nneuronDelta ="<<neuronDelta;
            }
        }
        delete[] errors;
    }
}

void NeuralNetwork::updateWeights(double* inputs, int dim, double lr) {
    if(isLogActive){
        cout<<"\n-update weights\n";
    }
    double* currentInputs = new double[dim];
    int currentInputsSize = dim;
    for(int i=0; i<dim; i++) {
        currentInputs[i] = inputs[i];
    }
    int numOfLayers = this->numOfLayers;
    for(int i=0; i<numOfLayers; i++) {
        Layer* layer = this->layers[i];
        int layerSize = layer->numOfNeurons;
        if(i!=0) {
            Layer* previousLayer = this->layers[i-1];
            int previuosLayerSize = previousLayer->numOfNeurons;
            delete[] currentInputs;
            currentInputs = new double[previuosLayerSize];
            currentInputsSize = previuosLayerSize;
            for(int j=0; j<previuosLayerSize; j++) {
                Neuron* neuron = previousLayer->neurons[j];
                currentInputs[j] = neuron->output;
            }
        }
        for(int n=0; n<layerSize; n++) {
            Neuron* neuron = layer->neurons[n];
            double* w = neuron->weights;
            double delta = neuron->delta;
            if(isLogActive) {
                cout<<"\nLayer "<<n;
            }
            for(int j=0; j<currentInputsSize; j++) {
                if(isLogActive) {
                    cout<<"\n w["<<j<<"]="<<w[j]<<" pre update";
                }
                double wUpdated = w[j] + lr*currentInputs[j]*delta;
                if(isLogActive) {
                    cout<<"\n w["<<j<<"]="<<wUpdated<<" post update\n";
                }
                w[j] += lr*currentInputs[j]*delta;
            }
            w[currentInputsSize] = lr*delta;
        }
    }
    delete[] currentInputs;
}

void NeuralNetwork::trainNetwork(const char* trainFileName, const char* weightsFileName, double lr, int numEpochs, int numOutputs) {
    if (isLogActive) {
        cout<<"\nTrain network by file - start\n";
    }
    FILE* trainFile = fopen(trainFileName, "r");
    if(trainFile == 0) {
        exit(EXIT_FAILURE);
    }
    
    int numSamples = 0;
    char line[MAX_LINE_LENGTH];

    int dimSample = -1;
    if (isLogActive) {
        cout<<"\nstart reading file...\n";
    }
    while(fgets(line, MAX_LINE_LENGTH, trainFile)) {
        numSamples++;
        if(dimSample == -1) {
            char* headToken = strtok(line, ",");
            while(headToken != NULL) {
                headToken = strtok(NULL, ",");
                dimSample++;
            }
        }
    }
    fclose(trainFile);
    if (isLogActive) {
        cout<<"\nfile infos read\n";
    }
    trainFile = fopen(trainFileName, "r");
    if(trainFile == 0) {
        exit(EXIT_FAILURE);
    }
    double** trainingSet = new double*[numSamples];
    numSamples = 0;
    while(fgets(line, MAX_LINE_LENGTH, trainFile)) {
        char* token = strtok(line, ",");
        double* trainRow = new double[dimSample+1];
        int i=0;
        while(token != NULL) {
            double sample = atof(token);
            trainRow[i] = sample;
            i++;
            token = strtok(NULL, ",");
        }
        trainingSet[numSamples] = trainRow;
        numSamples++;
    }
    fclose(trainFile);
    if (isLogActive) {
        cout<<"\ntraining set filled\n";
    }
    this->trainNetwork(trainingSet, weightsFileName, numSamples, dimSample, lr, numEpochs, numOutputs);
    this->deleteTrainingSet(trainingSet, numSamples);
    if (isLogActive) {
        cout<<"\nTrain network by file - end\n";
    }
}

void NeuralNetwork::trainNetwork(double** trainingSet, const char* weightsFileName, int numSamples, int dimSample, double lr, int numEpochs, int numOutputs) {
    for(int epoch=0; epoch<numEpochs; epoch++) {
        double sumError = 0.0;
        for(int i=0; i<numSamples; i++) {
            double* row = trainingSet[i];
            double* inputs = new double[dimSample];
            for(int j=0; j<dimSample; j++) {
                inputs[j] = row[j];
            }
            double* outputs = this->frowardPropagate(inputs, dimSample);
            double* expected = new double[numOutputs];
            for(int j=0; j<numOutputs; j++) {
                expected[j] = 0.0;
            }
            expected[(int)row[dimSample]] = 1.0;
            for(int k=0; k<numOutputs; k++) {
                sumError += pow((expected[k] - outputs[k]), 2);
            }
            this->backPropagate(expected, numOutputs);
            this->updateWeights(inputs, dimSample, lr);
            delete[] expected;
            delete[] inputs;
        }
        if (isLogActive) {
            cout << "\nepoch = " << epoch << ", learning rate = " << lr << ", error = " << sumError << "\n";
        }
    }
    this->saveNetwork(weightsFileName);
}

int NeuralNetwork::fit(double* inputs, int dim) {
    double* outputs = this->frowardPropagate(inputs, dim);
    Layer* outputLayer = this->layers[this->numOfLayers-1];
    int numOutputs = outputLayer->numOfNeurons;
    Neuron* neuron = outputLayer->neurons[0];
    double maxOut = neuron->output;
    for(int i=0; i<numOutputs; i++) {
        neuron = outputLayer->neurons[i];
        double neuronOut = neuron->output;
        if(neuronOut > maxOut) {
            maxOut = neuronOut;
        }
    }
    int res = -1;
    for(int i=0; i<numOutputs; i++) {
        if(outputs[i] == maxOut) {
            res = i;
        }
    }
    return res;
}

void NeuralNetwork::deleteTrainingSet(double** trainingSet,int dim) {
    for(int i=0; i<dim; i++) {
        double* trainRow = trainingSet[i];
        delete[] trainRow;
    }
    delete[] trainingSet;
}

void NeuralNetwork::saveNetwork(const char* fileName) {
    /*
     struttura file pesi:
     .
     .
     Layer i
     Neuron 1
     45.4
     .
     .
     Neuron n
     .
     .
     Layer 2
     .
     .
     Layer l
     .
     .
     */
    FILE* net = fopen(fileName, "w");
    char* layerNum = strdup("0");
    char* neuronNum = strdup("0");
    for(int l=0; l<this->numOfLayers; l++) {
        Layer* layer = this->layers[l];
        sprintf(layerNum, "%d", l);
        char* layerLabel = strdup(LAYER);
        strcat(layerLabel, layerNum);
        fprintf(net,"%s\n",layerLabel);
        for(int n=0; n<layer->numOfNeurons; n++) {
            char* neuronLabel = strdup(NEURON);
            sprintf(neuronNum, "%d", n);
            strcat(neuronLabel, neuronNum);
            fprintf(net,"%s\n",neuronLabel);
            Neuron* neuron = layer->neurons[n];
            for(int i=0; i<neuron->numOfInputs+1; i++) {
                fprintf(net,"%f\n",neuron->weights[i]);
            }
        }
    }
    fclose(net);
}
