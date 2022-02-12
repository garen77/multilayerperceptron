//
//  NeuralNetwork.hpp
//  NeuralNetwork
//
//  Created by Crescenzo Garofalo on 01/02/22.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>



using std::cout;

struct Neuron {
    double delta;
    double* weights;
    double output;
    int numOfInputs;
    
    Neuron(double dlt, double* w, double o, int ni) :delta(dlt), weights(w), output(o), numOfInputs(ni) {}
    ~Neuron() {
        delete[] weights;
    }
};

struct Layer {
    Neuron** neurons;
    int numOfNeurons;
    
    Layer() {}
    ~Layer() {
        for(int i=0; i<numOfNeurons; i++) {
            delete neurons[i];
        }
        delete[] neurons;
    }
};

class NeuralNetwork {

private:
    Layer** layers;
    int numOfLayers;
    int* configuration;
    
public:
    NeuralNetwork(int* conf, int nl);
    NeuralNetwork(const char* weightsFileName);
    ~NeuralNetwork();
    double activate(Neuron* neuron, double* inputs);
    double* frowardPropagate(double* inputs, int dim);
    void backPropagate(double* expected, int dim);
    void updateWeights(double* inputs, int dim, double lr);
    void trainNetwork(const char* trainFileName, const char* weightsFileName, double lr, int numEpochs, int numOutputs);
    void trainNetwork(double** trainingSet, const char* weightsFileName, int numSamples, int dimSample, double lr, int numEpochs, int numOutputs);
    int fit(double* inputs, int dim);
    void deleteTrainingSet(double** trainingSet,int dim);
    
    void saveNetwork(const char* fileName);
};

#endif /* NeuralNetwork_hpp */
