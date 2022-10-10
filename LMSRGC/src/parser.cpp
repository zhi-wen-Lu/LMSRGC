#include "parser.h"

int parse(FILE *stream, vector<string>& taxa, vector<string>& sequences){
	vector<char> seqBuffer;
	vector<char> taxaBuffer;
	char c=getc(stream);
	while(c!=EOF){
		if(c=='>'){
			while((c=getc(stream))!='\n'){
				taxaBuffer.push_back(c);		
			}
		}
		else if(c==';')
			while((c=getc(stream))!='\n');
		else{
			while((c=getc(stream))!='>' && c!=EOF){
				if(!isalpha(c)){
					continue;
				}
				c=toupper(c);
				seqBuffer.push_back(c);	
			}	
		}
		if(seqBuffer.size()>0){
			string taxaString(taxaBuffer.begin(),taxaBuffer.end());
			string seqString(seqBuffer.begin(),seqBuffer.end());
			taxa.push_back(taxaString);
			sequences.push_back(seqString);
			seqBuffer.clear();
			taxaBuffer.clear();
		}
	}
	return FASTA_OK;
}

int writeDmat(double** DMat, vector<string>& taxa){
	ofstream outfile;
	outfile.open("DMat");
	outfile << taxa.size() <<endl;
		for(int i=0;i<taxa.size();i++){
			outfile << left << setw(10) << taxa.at(i).substr(0,10);
				for(int j=0;j<taxa.size();j++){
					if(i!=j){
						outfile << setprecision(8) << DMat[j][i] << "  ";
						}
					else
						outfile << setprecision(8) << "0" << "  ";
				}
			outfile << endl;
		}
	//outfile << DMat[0][1] <<endl;
	outfile.close();
	return FASTA_OK;
}
