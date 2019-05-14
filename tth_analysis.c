void tth_analysis(){

TChain *chain1= new  TChain("ttH_0M_125_13TeV_TTHHadronicTag");
TChain *chain2= new  TChain("ttH_0P_125_13TeV_TTHHadronicTag");

chain1->Add("ttHiggs0MToGG.root");
chain2->Add("ttHiggs0PToGG.root");
int var_num=6,entry_num1,entry_num2;
entry_num1=chain1->GetEntries();
entry_num2=chain2->GetEntries();
cout<<"entry_num1="<<entry_num1<<endl;
cout<<"entry_num2="<<entry_num2<<endl;
string delta_phipho,delta_phijet;
delta_phipho="(abs(pho1_phi-pho2_phi))*(abs(pho1_phi-pho2_phi)<3.14)+\
(2*3.14-abs(pho1_phi-pho2_phi))*(abs(pho1_phi-pho2_phi)>3.14)";
delta_phijet="(abs(jetPhi_1-jetPhi_2))*(abs(jetPhi_1-jetPhi_2)<3.14)+\
(2*3.14-abs(jetPhi_1-jetPhi_2))*(abs(jetPhi_1-jetPhi_2)>3.14)";

TString v[]={"pho1_eta-jetEta_1",delta_phipho,
"pho1_phi-jetPhi_1",delta_phijet,"jetEta_1-jetEta_2","pho1_eta-pho2_eta"};
float xlow[]={-5,0,-4,0,-5.0,-5.0};
float xup[]={5,3.5,4,3.5,5.0,5.0};
TString title[]={"#Delta #eta_{#gamma_{1} j_{1}} ",
 "#Delta #phi_{#gamma_{1} #gamma_{2}}","#Delta #phi_{#gamma_{1} j_{1}}",
 "#Delta #phi_{j_{1}j_{2}} ","#Delta #eta_{j_{1}j_{2}}",
"#Delta #eta_{#gamma_{1} #gamma_{2}} ","MET"};
for(int i=0;i< var_num;i++) 
	{ 
	auto canvas= new TCanvas();// not necessary
  TH1F *h1=new TH1F("ptm",title[i]+ " distribution under CP odd or even",
  	100,xlow[i],xup[i]);
  chain1->Draw(v[i]+">>ptm","weight");
  h1->SetLineColor(3);
  h1->SetStats(kFALSE); //eliminate the sta  
  h1->GetXaxis()->SetTitle(title[i]);
  h1->GetYaxis()->SetTitle("Rates");  

  TH1F *h2=new TH1F("ptp",title[i]+" distribution under CP odd or even ",
  	100,xlow[i],xup[i]);
    chain2-> Draw(v[i]+">>ptp","weight");
	h2->SetLineColor(4);
	h2->SetStats(kFALSE);
	h2->GetXaxis()->SetTitle(title[i]);
  	h2->GetYaxis()->SetTitle("Rates"); 


	double maxh1,maxh2;
	maxh1=h1->GetMaximum();
	maxh2=h2->GetMaximum();
	if(maxh1/entry_num1 > maxh2/entry_num2){
		h1->DrawNormalized("Hist");
		h2->DrawNormalized("HistSame");
	}
	else{h2->DrawNormalized("hist");
		h1->DrawNormalized("histSame");
	}

	TLegend leg(.7,.7,.9,.9);
	leg.SetFillColor(0);
	leg.AddEntry(h1,"CP Odd");
	leg.AddEntry(h2,"CP Even");
	leg.DrawClone("Same");

	 gPad->Print("plot"+title[i]+".png");
   //  gPad->SaveAs("plot"+v[i]+".png")

	}

}



/*	TFile *fm=new TFile("ttHiggs0MToGG.root");
	TFile *fp=new TFile("ttHiggs0PToGG.root");
	TTree *tm, *tp;
	fm->GetObject("ttH_0M_125_13TeV_TTHHadronicTag",tm);
	fp->GetObject("ttH_0P_125_13TeV_TTHHadronicTag",tp);
	TCanvas *c=new TCanvas("c","p_{t} didstribution ofjet1 ");
	TH1F *h1=new TH1F("ptm","jet pt under CP odd",100,0,2000);
	TH1F *h2=new TH1F("ptp","jet pt under CP even",100,0,2000);
	Float_t jetPt1m,jetPt1p,weight1,weight2;
	tm->SetBranchAddress("jetPt_1",&jetPt1m);
	tp->SetBranchAddress("jetPt_1",&jetPt1p);
	tm->SetBranchAddress("weight",&weight1);
	tp->SetBranchAddress("weight",&weight2);

	for(int i=0;i< tm->GetEntries();i++) 
		{tm->GetEntry(i);h1->Fill(jetPt1m,weight1);}
	for(int i=0;i< tp->GetEntries();i++) 
		{tp->GetEntry(i);h2->Fill(jetPt1p,weight2);}
	c->cd();
	h1->Draw("hist");
	h2->Draw("histSame");
	h2->SetTitle();
*/
