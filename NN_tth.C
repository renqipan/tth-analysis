/*TMVA Classification Example Using a Convolutional Neural Network
Declare Factory
Create the Factory class. Later you can choose the methods whose performance you'd like to investigate.

The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to pass

The first argument is the base of the name of all the output weightfiles in the directory weight/ that will be created with the method parameters

The second argument is the output file for the training results

The third argument is a string option defining some general configuration for the TMVA session. 
For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in the option string
*/


#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

void NN_tth(){

//Declare Factory
TMVA::Tools::Instance();
auto sigFile = TFile::Open("./ttHiggs0PToGG.root");
auto bkgFile=TFile::Open("./ttHiggs0MToGG.root");

 // for using Keras
gSystem->Setenv("KERAS_BACKEND","tensorflow"); 
TMVA::PyMethodBase::PyInitialize();


TString outfileName("NN_tth_Classification.root");
auto outputFile = TFile::Open(outfileName, "RECREATE");
TMVA::Factory factory("NN_tth_Classification", outputFile,
                      "!V:ROC:!Silent:Color:!DrawProgressBar:AnalysisType=Classification" );

TString delta_phi_pho,delta_phi_jet;
delta_phi_pho="delta_phi_pho:=(abs(pho1_phi-pho2_phi))*(abs(pho1_phi-pho2_phi)<3.14)+\
(2*3.14-abs(pho1_phi-pho2_phi))*(abs(pho1_phi-pho2_phi)>3.14)";
delta_phi_jet="delta_phi_jet:=(abs(jetPhi_1-jetPhi_2))*(abs(jetPhi_1-jetPhi_2)<3.14)+\
(2*3.14-abs(jetPhi_1-jetPhi_2))*(abs(jetPhi_1-jetPhi_2)>3.14)";

//Declare DataLoader(s)
TMVA::DataLoader loader("dataset");
loader.AddVariable("delta_phoj_eta:=pho1_eta-jetEta_1",'F');
loader.AddVariable("delta_phoj_phi:=pho1_phi-jetPhi_1",'F');
loader.AddVariable("delata_jj_eta:=jetEta_1-jetEta_2",'F');
loader.AddVariable("delta_pho_eta:=pho1_eta-pho2_eta",'F');
loader.AddVariable(delta_phi_pho,"delta_phi_pho","",'F');
loader.AddVariable(delta_phi_jet,"delta_phi_jet","",'F');

//Setup Dataset(s)
TTree *tsignal, *tbackground;
sigFile->GetObject("ttH_0P_125_13TeV_TTHHadronicTag", tsignal);
bkgFile->GetObject("ttH_0M_125_13TeV_TTHHadronicTag", tbackground);

TCut mycuts, mycutb;

Double_t signalWeight     = 1.0;
Double_t backgroundWeight = 1.0;
loader.AddSignalTree    (tsignal,     signalWeight);   //signal weight  = 1
loader.AddBackgroundTree(tbackground, backgroundWeight);   //background weight = 1 

loader.SetBackgroundWeightExpression( "weight" ); //Set individual event weights 
loader.SetSignalWeightExpression("weight");
loader.PrepareTrainingAndTestTree(mycuts, mycutb,
                                   "nTrain_Signal=30000:nTrain_Background=30000:SplitMode=Random:NormMode=NumEvents:!V" );
//Booking Methods
//Boosted Decision Trees
factory.BookMethod(&loader,TMVA::Types::kBDT, "BDT",
                   "!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

//Booking  Neural Network
//Here we book the new DNN of TMVA. If using master version you can use the new DL method

bool useDNN = true; 
bool useCNN = true; 
bool useKeras =false;

if (useDNN) { 
    
     TString layoutString ("Layout=TANH|128,TANH|128,TANH|128,LINEAR");

      // Training strategies.
      TString training0("LearningRate=1e-1,Momentum=0.9,Repetitions=1,"
                        "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                        "WeightDecay=1e-4,Regularization=L2,"
                        "DropConfig=0.0+0.5+0.5+0.5, Multithreading=True");
      TString training1("LearningRate=1e-2,Momentum=0.9,Repetitions=1,"
                        "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                        "WeightDecay=1e-4,Regularization=L2,"
                        "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
      TString training2("LearningRate=1e-3,Momentum=0.0,Repetitions=1,"
                        "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                        "WeightDecay=1e-4,Regularization=L2,"
                        "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
      TString trainingStrategyString ("TrainingStrategy=");
      trainingStrategyString += training0 + "|" + training1 + "|" + training2;

      // General Options.                                                                                                                                                                
      TString dnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:"
                          "WeightInitialization=XAVIERUNIFORM");
      dnnOptions.Append (":"); dnnOptions.Append (layoutString);
      dnnOptions.Append (":"); dnnOptions.Append (trainingStrategyString);

      dnnOptions += ":Architecture=CPU";
      factory.BookMethod(&loader, TMVA::Types::kDNN, "DNN_CPU", dnnOptions);

}

//Book Convolutional Neural Network in Keras using a generated model
if (useKeras) { 
   factory.BookMethod(&loader, TMVA::Types::kPyKeras, 
                       "PyKeras","H:!V:VarTransform=None:FilenameModel=model_cnn.h5:"
                       "FilenameTrainedModel=trained_model_cnn.h5:NumEpochs=20:BatchSize=256");
}

//Train Methods
factory.TrainAllMethods();
//Test and Evaluate Methods
factory.TestAllMethods();
factory.EvaluateAllMethods();
outputFile->Close(); // Save the output

//Plot ROC Curve
auto c1 = factory.GetROCCurve(&loader);
c1->Draw();
gPad->Print("Significance.png");
  // Launch the GUI for the root macros
if (!gROOT->IsBatch()) TMVA::TMVAGui(outfileName);

}