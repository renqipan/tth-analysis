//Toy Monte carlo example
//check pull distribution to compare chi and binned log-likelihood methods.
void pull(int n_toys=10000, int n_tot_entries=100,int nbins=40,bool do_chi2=true){
	TString method_prefix("log-likelihood");
	if(do_chi2) method_prefix="#chi^{2}";

	//create histo
	TH1F *h4=new TH1F(method_prefix+"h4",method_prefix+" Random Gauss",nbins,-4,4);
	h4->SetMarkerStyle(21);
	h4->SetMarkerSize(0.8);
	h4->SetMarkerColor(kRed);

	//histogram for sigma and pull
	TH1F *sigma=new TH1F(method_prefix+"sigma",method_prefix+" sigma for Gauss fit",50,0.5,1.5);
	TH1F *pull=new TH1F(method_prefix+"pull",method_prefix+" pull from gaus fit",50,-4,4);

	//make nice canvases
	TCanvas *c0=new TCanvas(method_prefix+"Gauss",method_prefix+" Gauss",0,0,500,500);
	c0->SetGrid();
	//make nice canvas
	TCanvas *c1=new TCanvas(method_prefix+"Result",method_prefix+" Pull Distribution",
		0,300,600,400);
	c0->cd();
	float sig , mean;
	for(int i=0;i<n_toys;i++){
		h4->Reset();
		for(int j=0;j<n_tot_entries;j++)
			h4->Fill(gRandom->Gaus());
		//perform fit
		if(do_chi2) h4->Fit("gaus","q");//chi2 fit
		else h4->Fit("gaus","lq");//likelihood hit
		if(!(i%100)){
			h4->Draw("ep");
			c0->Update();
		}
		TF1 *fit=h4->GetFunction("gaus");
		sig=fit->GetParameter(2);
		mean=fit->GetParameter(1);
		sigma->Fill(sig);
		pull->Fill(mean/sig *sqrt(n_tot_entries));
	}   //end of toy MC loop
	//print result
	c1->cd();
	pull->Draw();
}

void macro9(){
	int n_tot_entries=100;
	int n_toys=10000;
	int n_bins=40;
	cout<<"perforing pull experiment witn ch2"<<endl;
	pull(n_toys,n_tot_entries,n_bins,true);
	cout<<"perforing pull experiment with log likehood"<<endl;
		pull(n_toys,n_tot_entries,n_bins,false);

}