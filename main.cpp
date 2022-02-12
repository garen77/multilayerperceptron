//
//  main.cpp
//  NeuralNetwork
//
//  Created by Crescenzo Garofalo on 01/02/22.
//

#include <iostream>
#include "NeuralNetwork.hpp"

// learning rate
#define RATE 0.7
#define MAX_LINE_LENGTH 1024

/*
 Use example:
 ./mnist 0 mnist_train.csv 7 50000 0.2 0.01 9 100
 */

int main(int argc, const char * argv[]) {
    
    // type 0: train - 1: test
    int type;
    NeuralNetwork* network;
    int o;
    if(argc < 3) {
        printf("\nuse: ./neural 0 <\"2 3 6 6 2\"> <train_file.csv> <network.w> <learn_rate> <epochs_num>\n");
        printf("\nor\n");
        printf("\nuse: ./neural 1 <test_file.csv> <network.w>\n");
    }
    
    if(sscanf(argv[1], "%i", &type) != 1) {
        printf("%s is not integer\n", argv[1]);
        exit(1);
    }

    if(type==0) {
        int epochs = 1;
        double lr = RATE;
        
        if(sscanf(argv[6], "%i", &epochs) != 1) {
            printf("%s is not integer\n", argv[6]);
            exit(1);
        }

        if(sscanf(argv[5], "%lf", &lr) != 1) {
            printf("%s is not double\n", argv[5]);
            exit(1);
        }

        u_long dimConfigArg = strlen(argv[2]);

        char* configArgv = (char *) malloc(dimConfigArg+1);
        //char configArgv[dimConfigArg];

        strcpy(configArgv, argv[2]);
        char* configItem = strtok(configArgv, " ");

        int confDim=0;
        while(configItem != NULL) {
            configItem = strtok(NULL, " ");
            confDim++;
        }
        int* configuration = new int[confDim];
        free(configArgv);
        configArgv = (char *) malloc(strlen(argv[2])+1);

        strcpy(configArgv, argv[2]);

        configItem = strtok(configArgv, " ");

        int index=0;
        while(configItem != NULL && index<confDim) {
            configuration[index] = atoi(configItem);
            index++;
            configItem = strtok(NULL, " ");
        }
        free(configArgv);
        network = new NeuralNetwork(configuration,confDim-1);
        cout<<"\n-----start training------\n";
        network->trainNetwork(argv[3], argv[4], lr, epochs, 2);
        cout<<"\n-----end training------\n";
        delete[] configuration;
        delete network;
        
    } else {
        
        network = new NeuralNetwork(/*"/Users/crescenzogarofalo/Documents/reti/NeuralNetwork/NeuralNetwork/xnor_weights.net"*/argv[3]);
    
        FILE* testFile = fopen(/*"/Users/crescenzogarofalo/Documents/reti/NeuralNetwork/NeuralNetwork/xnor_test.csv"*/argv[2], "r");
        char line[MAX_LINE_LENGTH];
        
        if(testFile == 0) {
            exit(EXIT_FAILURE);
        }
        std::vector<std::vector<double>*>* testRows = new std::vector<std::vector<double>*>();
        std::vector<double>* testRow;
        int dimTestRow=0;
        bool isFirst = true;
        while(fgets(line, MAX_LINE_LENGTH, testFile) != NULL) {
            testRow = new std::vector<double>();
            char* headToken = strtok(line, ",");
            while(headToken != NULL) {
                testRow->push_back(atof(headToken));
                if(isFirst) {
                    dimTestRow++;
                }
                headToken = strtok(NULL, ",");
            }
            isFirst = false;
            testRows->push_back(testRow);
        }
        double* inps;
        int numOfRows = (int)testRows->size();
        for(int i=0; i<numOfRows; i++) {
            testRow = testRows->at(i);
            int sampleDim = (int)testRow->size()-1;
            inps = new double[sampleDim];
            for(int j=0; j<sampleDim; j++) {
                inps[j] = testRow->at(j);
            }
            int expected = testRow->at(sampleDim);
            o = network->fit(inps, sampleDim);
            cout<<"\nFit\nexpected = "<<expected<<" out = "<<o<<"\n";
            delete[] inps;
        }
        for(int i=0; i<numOfRows; i++) {
            testRow = testRows->at(i);
            delete testRow;
        }
        delete testRows;
    }
    
    /*int* conf; = new int[3] {2,2,2};
     double** trainingSet = new double*[31];
    trainingSet[0] = new double[] {2.7810836,2.550537003,0};
    trainingSet[1] = new double[] {1.465489372,2.362125076,0};
    trainingSet[2] = new double[] {3.396561688,4.400293529,0};
    trainingSet[3] = new double[] {1.38807019,1.850220317,0};
    trainingSet[4] = new double[] {3.06407232,3.005305973,0};
    trainingSet[5] = new double[] {7.627531214,2.759262235,1};
    trainingSet[6] = new double[] {5.332441248,2.088626775,1};
    trainingSet[7] = new double[] {6.922596716,1.77106367,1};
    trainingSet[8] = new double[] {8.675418651,-0.242068655,1};
    trainingSet[9] = new double[] {8.675418651,-0.242068655,1};
    trainingSet[10] = new double[] {7.673756466,3.508563011,1};
    trainingSet[11] = new double[] { 2.7810836,2.550537003,0 };
    trainingSet[12] = new double[] { 1.465489372,2.362125076,0 };
    trainingSet[13] = new double[] { 3.396561688,4.400293529,0 };
    trainingSet[14] = new double[] { 1.38807019,1.850220317,0 };
    trainingSet[15] = new double[] { 3.06407232,3.005305973,0 };
    trainingSet[16] = new double[] { 7.627531214,2.759262235,1 };
    trainingSet[17] = new double[] { 5.332441248,2.088626775,1 };
    trainingSet[18] = new double[] { 6.922596716,1.77106367,1 };
    trainingSet[19] = new double[] { 8.675418651,-0.242068655,1 };
    trainingSet[20] = new double[] { 7.673756466,3.508563011,1 };
    trainingSet[21] = new double[] { 2.7810836,2.550537003,0 };
    trainingSet[22] = new double[] { 1.465489372,2.362125076,0 };
    trainingSet[23] = new double[] { 3.396561688,4.400293529,0 };
    trainingSet[24] = new double[] { 1.38807019,1.850220317,0 };
    trainingSet[25] = new double[] { 3.06407232,3.005305973,0 };
    trainingSet[26] = new double[] { 7.627531214,2.759262235,1 };
    trainingSet[27] = new double[] { 5.332441248,2.088626775,1 };
    trainingSet[28] = new double[] { 6.922596716,1.77106367,1 };
    trainingSet[29] = new double[] { 8.675418651,-0.242068655,1 };
    trainingSet[30] = new double[] { 7.673756466,3.508563011,1 };

    network = new NeuralNetwork(conf,2);
    
    cout << "\nFirst train\n";
    network->trainNetwork(trainingSet, 31, 2, 0.5, 40, 2);
    
    inps = new double[] {6.922596716,1.77106367};
    o = network->fit(inps, 2);
    cout<<"\nFit\nexpected = 1 out = "<<o<<"\n";
    delete[] inps;
    inps = new double[] {3.396561688,4.400293529};
    o = network->fit(inps, 2);
    cout<<"\nFit\nexpected = 0 out = "<<o<<"\n";
    delete[] inps;
    
    delete[] conf;
    network->deleteTrainingSet(trainingSet,31);*/
    
    /*cout<<"\nTrain - xor\n";
    
    int* conf = new int[5];
    conf[0] = 2;
    conf[1] = 4;
    conf[2] = 6;
    conf[3] = 6;
    conf[4] = 2;
    
    network = new NeuralNetwork(conf,4);
    network->trainNetwork("/Users/crescenzogarofalo/Documents/reti/NeuralNetwork/NeuralNetwork/xor_train.csv","/Users/crescenzogarofalo/Documents/reti/NeuralNetwork/NeuralNetwork/xor_weights.net" , 0.7, 2000, 2);

    double* inps = new double[2];
    inps[0] = 0;
    inps[1] = 1;
    o = network->fit(inps, 2);
    cout<<"\nFit - xor\nexpected = 1 out = "<<o<<"\n";
    delete[] inps;
    
    inps = new double[2];
    inps[0] = 1;
    inps[1] = 0;
    o = network->fit(inps, 2);
    cout<<"\nFit - xor\nexpected = 1 out = "<<o<<"\n";
    delete[] inps;
    
    inps = new double[2];
    inps[0] = 1;
    inps[1] = 1;
    o = network->fit(inps, 2);
    cout<<"\nFit - xor\nexpected = 0 out = "<<o<<"\n";
    delete[] inps;
    
    inps = new double[2];
    inps[0] = 0;
    inps[1] = 0;
    o = network->fit(inps, 2);
    cout<<"\nFit - xor\nexpected = 0 out = "<<o<<"\n";
    delete[] inps;

    delete[] conf;*/
    
    /*cout << "\nTrain - xnor\n";
    
    conf = new int[5]{ 2,6,6,6,2 };
    network = new NeuralNetwork(conf,4);
    network->trainNetwork("/Users/crescenzogarofalo/Documents/reti/NeuralNetwork/NeuralNetwork/xnor_train.csv", 0.7, 2000, 2);
    
    inps = new double[] {0,1};
    o = network->fit(inps, 2);
    cout<<"\nFit - xnor\nexpected = 0 out = "<<o<<"\n";
    delete[] inps;
    
    inps = new double[] {1,0};
    o = network->fit(inps, 2);
    cout<<"\nFit - xnor\nexpected = 0 out = "<<o<<"\n";
    delete[] inps;
    
    inps = new double[] {1,1};
    o = network->fit(inps, 2);
    cout<<"\nFit - xnor\nexpected = 1 out = "<<o<<"\n";
    delete[] inps;
    
    inps = new double[] {0,0};
    o = network->fit(inps, 2);
    cout<<"\nFit - xnor\nexpected = 1 out = "<<o<<"\n";
    delete[] inps;*/

    
    cout<<"\n";
    return 0;
}
