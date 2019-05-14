//this is a c++ program for tth analysis application using trained and tested result
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

void BDT_tth_Application(){
	// This loads the library
    TMVA::Tools::Instance();
	TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );
	Float_t delta_phoj_eta,delta_phoj_phi,delata_jj_eta,delta_pho_eta;
	Float_t	mydelta_phi_pho,mydelta_phi_jet;
	TString delta_phi_pho,delta_phi_jet;
	delta_phi_pho="delta_phi_pho:=(abs(pho1_phi-pho2_phi))*(abs(pho1_phi-pho2_phi)<3.14)+\
(2*3.14-abs(pho1_phi-pho2_phi))*(abs(pho1_phi-pho2_phi)>3.14)";
	delta_phi_jet="delta_phi_jet:=(abs(jetPhi_1-jetPhi_2))*(abs(jetPhi_1-jetPhi_2)<3.14)+\
(2*3.14-abs(jetPhi_1-jetPhi_2))*(abs(jetPhi_1-jetPhi_2)>3.14)";

	reader->AddVariable("delta_phoj_eta:=pho1_eta-jetEta_1",&delta_phoj_eta);
	reader->AddVariable("delta_phoj_phi:=pho1_phi-jetPhi_1",&delta_phoj_phi);
	reader->AddVariable("delata_jj_eta:=jetEta_1-jetEta_2",&delata_jj_eta);
	reader->AddVariable("delta_pho_eta:=pho1_eta-pho2_eta",&delta_pho_eta);
	reader->AddVariable(delta_phi_pho,&mydelta_phi_pho);
	reader->AddVariable(delta_phi_jet,&mydelta_phi_jet);

	TString BDTFileName="TMVAClassification_BDT.weights.xml";
	TString MLPFileName="TMVAClassification_MLP.weights.xml";
	TString dir="/home/renqi/Documents/root_practice/dataset/weights/";
	TString BTDFile=dir+BDTFileName;
	TString MLPFile=dir+MLPFileName;
	reader->BookMVA("BDT method", BTDFile );
	reader->BookMVA("MLP method",MLPFile);
	int nbin=100;
    TH1F *histBDT= new TH1F( "MVA_BDT","MVA_BDT",nbin, -0.8,0.8 );
 	TH1F *histMLP= new TH1F( "MVA_MLP","MVA_MLP",nbin, -0.8, 0.8 );
 	TString inputFile="./ttHiggs0PToGG.root";
	auto sigFile = TFile::Open(inputFile);
	TTree *tsignal;
	sigFile->GetObject("ttH_0P_125_13TeV_TTHHadronicTag", tsignal);

	Float_t pho1_phi,pho2_phi,jetPhi_1,jetPhi_2;
	Float_t pho1_eta,pho2_eta,jetEta_1,jetEta_2;
	tsignal->SetBranchAddress("pho1_eta",&pho1_eta);
	tsignal->SetBranchAddress("pho2_etao",&pho2_eta);
	tsignal->SetBranchAddress("jetEta_2",&jetEta_2);
	tsignal->SetBranchAddress("jetEta_1",&jetEta_1);
	tsignal->SetBranchAddress("jetPhi_1",&jetPhi_1);
	tsignal->SetBranchAddress("jetPhi_2",&jetPhi_2);


	std::cout << "--- Processing: " << tsignal->GetEntries() << " events" << std::endl;
   TStopwatch sw;
   sw.Start();

	for (Long64_t ievt=0; ievt<tsignal->GetEntries();ievt++) {

	      if (ievt%1000 == 0) std::cout << "--- ... Processing event: " << ievt << std::endl;

	      tsignal->GetEntry(ievt);
	      delta_phoj_eta=pho1_eta-jetEta_1;
	      delta_phoj_phi=pho1_phi-jetPhi_1;
	      delata_jj_eta=jetEta_1-jetEta_2;
	      delta_pho_eta=pho1_eta-pho2_eta;
	      mydelta_phi_pho=(abs(pho1_phi-pho2_phi))*(abs(pho1_phi-pho2_phi)<3.14)+
						(2*3.14-abs(pho1_phi-pho2_phi))*(abs(pho1_phi-pho2_phi)>3.14);
		  mydelta_phi_jet=(abs(jetPhi_1-jetPhi_2))*(abs(jetPhi_1-jetPhi_2)<3.14)+
			(2*3.14-abs(jetPhi_1-jetPhi_2))*(abs(jetPhi_1-jetPhi_2)>3.14);

		histBDT ->Fill( reader->EvaluateMVA( "BDT method") );
		histMLP ->Fill( reader->EvaluateMVA( "MLP method") );
	}
	// Get elapsed time
   sw.Stop();
   std::cout << "--- End of event loop: "; sw.Print();
    TFile *target  = new TFile( "tth_BDT.root","RECREATE" );
    histMLP->Write();
    histBDT->Write();

    target->Close();
	std::cout << "--- Created root file: \"tth_BDT.root\" containing the MVA output histograms" << std::endl;

   delete reader;

   std::cout << "==> TMVAClassificationApplication is done!" << std::endl << std::endl;

}