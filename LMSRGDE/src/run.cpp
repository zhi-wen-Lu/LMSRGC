

#include <ctype.h>

#include <stdio.h>

#include <stdlib.h>

#include <unistd.h>

#include <iostream>

#include <math.h>

#include <stdlib.h>

#include <cstdio>

#include <fstream>

#include <string>

#include <vector>

#include <cmath>

#include <ctime>

#include <list>

#include <iomanip>

#include <string.h>

#include "sais.h"

#include "libbsc/bsc.h"

using namespace std;

int Ref_len,compress_len,tar_len,Ver_len;

int max = 1<<30;

char *Ref_seq;

char *Tar_seq;

char *Ver_seq;

char *Comp_seq;

char *temp;

const int min_size = 1<<23;

int pos_vec_len;

static const char alphanum[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

inline std::string generateString(const std::string &chr, int length = 5) {

	srand(time(0));

	string res = chr + "_";

	for (int i = 0; i < length; ++i) {

		res += alphanum[rand() % 62];

	}

	return res;

}

struct POSITION_RANGE{

	int begin, length;

};

POSITION_RANGE *pos_vec = new POSITION_RANGE[min_size];

void readReflow(FILE *Rstream, vector<string>&Refsequence,vector<string> &meta){

	vector<char> seqBuffer;

	vector<char> taxaBuffer;

	int n_len = 0;

	char c=getc(Rstream);

	int ref_len = 0;

	while(c!=EOF){

		if(c=='>'||c == '@'){

			taxaBuffer.push_back(c);

			while((c=getc(Rstream))!='\n'){

				taxaBuffer.push_back(c);		

			}

		}

		else{

			while((c=getc(Rstream))!='>' && c!=EOF){

				if(!isalpha(c)){

					continue;

				}

				//c=toupper(c);

				//seqBuffer.push_back(c);

				Ref_seq[ref_len] = c;

				ref_len ++;

			

			}	

		}

	}

		int rc_len = ref_len;

	if(rc_len >= 0)

	{

		char ch;

		while (rc_len >= 0)

		{



			switch(Ref_seq[rc_len]){

				case 'A':

					ch = 'T';

					Ref_seq[ref_len++] = ch;

					break;

				case 'C':

					ch = 'G';

					Ref_seq[ref_len++] = ch;

					break;

				case 'G':

					ch = 'C';

					Ref_seq[ref_len++] = ch;

					break;

				case 'T':

					ch = 'A';

					Ref_seq[ref_len++] = ch;

					break;

				case 'a':

					ch = 't';

					Ref_seq[ref_len++] = ch;

					break;

				case 'c':

					ch = 'g';

					Ref_seq[ref_len++] = ch;

					break;

				case 'g':

					ch = 'c';

					Ref_seq[ref_len++] = ch;

					break;

				case 't':

					ch = 'a';

					Ref_seq[ref_len++] = ch;

					break;

				break;

			}

			rc_len --;

		}

		

	}

	Ref_seq[ref_len] = '\0';

	string taxaString(taxaBuffer.begin(),taxaBuffer.end());

	meta.push_back(taxaString);

	taxaBuffer.clear();

	

	return ;

}

void readRef(FILE *Rstream, vector<string>&Refsequence,vector<string> &meta){

	vector<char> seqBuffer;

	vector<char> taxaBuffer;

	int n_len = 0,ref_len = 0;

	char c=getc(Rstream);

	

	while(c!=EOF){

		

		if(c=='>'||c == '@'){

			

			taxaBuffer.push_back(c);

			while((c=getc(Rstream))!='\n'){

				taxaBuffer.push_back(c);		

			}

		

		}else{

			

			while((c=getc(Rstream))!='>' && c!=EOF){

				if(!isalpha(c)){

					continue;

				}

				c=toupper(c);

				//seqBuffer.push_back(c);

				Ref_seq[ref_len++] = c;

			

			}

		

		}

	}

	int rc_len = ref_len;

	if(rc_len >= 0)

	{

		char ch;

		while (rc_len >= 0)

		{

			switch(Ref_seq[rc_len]){

				case 'A':

					ch = 'T';

					Ref_seq[ref_len++] = ch;

					break;

				case 'C':

					ch = 'G';

					Ref_seq[ref_len++] = ch;

					break;

				case 'G':

					ch = 'C';

					Ref_seq[ref_len++] = ch;

					break;

				case 'T':

					ch = 'A';

					Ref_seq[ref_len++] = ch;

					break;

				break;

			}

			rc_len --;

		}

		

	}

	Ref_seq[ref_len] = '\0';

		

	string taxaString(taxaBuffer.begin(),taxaBuffer.end());

			

	meta.push_back(taxaString);

	

	return ;

}



void outputdecode(FILE* Gstream,char *Tar_seq,char *decode,vector<string> &meta){

	char *temp_seq = new char[ 1 << 28];int temp_seq_len = 0;

	for(int z = 0;z < tar_len;z ++){

		if(z %60 == 0){

			temp_seq[temp_seq_len ++] = '\n';

		}

		temp_seq[temp_seq_len ++] = Tar_seq[z];

	}

	string meta_deq;

	meta_deq =meta.at(0);

	std::ofstream fn;

	string output;

	output = decode;

	fn.open(output);

	fn<< meta_deq << temp_seq;

	delete[] temp_seq;

	//delete[] meta;

	meta.clear();

}



void readTar(FILE *Tstream){

	vector<char> taxaBuffer;

	char c = getc(Tstream);

	int _ver_len = 0;

	while(c != EOF){

		if(c == '>'|| c == '@'){

			while ((c = getc(Tstream))!= '\n')

			{

				taxaBuffer.push_back(c);

			}

		}

		else{

			while ((c = getc(Tstream)) != '>' && c != EOF)

			{

				if(!isalpha(c)){

					continue;

				}

				Ver_seq[_ver_len++] = c;

				if(_ver_len ==13570){

					int ww =0;

				}

			}

			

		}

	}

	Ver_len =_ver_len;

}



void Verification(char *Ver_seq,char *Tar_seq){

	int tot = 0;

	if(tar_len == Ver_len){

		for(int y = 0; y < Ver_len; y++){ 

			if(Tar_seq[y] == Ver_seq[y]){

				tot ++;

			}else if(Tar_seq[y] != Ver_seq[y]){

				int m = y;

				cout << m << endl;

			}

		}

		cout << "success" << endl;

	}else if(tar_len != Ver_len){

		for(int y = 0; y < Ver_len; y++){

			if(Tar_seq[y] == Ver_seq[y]){

				tot++;

			}else if(Tar_seq[y] != Ver_seq[y]){

				int m = y;

				cout << "error" << endl;

				cout << m << endl;

			}

		}

	}

	int ww = tot;

}



void readCompressfile(FILE *Cstreaam){

	int _com_len = 0;

	char c;

	while((c = getc(Cstreaam)) != EOF){

		Comp_seq[_com_len ++] = c;

	}

	compress_len = _com_len;



}



void decodeCompress(FILE *Gstream,FILE *Cstream, FILE *Rstream,vector<string> &meta){//vector<string>&Refsequence

	int i = 0;char bc[1 << 22];int curpos;int prepos = 0;int samelen;int diflen;

	const char eolChar1 = 10;const char eolChar2 = 13;int totlen = 0;tar_len = 0;int uplen = 0;

	bool flag_mis = false;int max = 1<< 28;

	int refpos =0;int mis_len =0;string ref;

	vector<string> Refsequence;

	

	/*ref = Refsequence.at(0);

	int reflength = Refsequence.at(0).length();

	unsigned char *Ref_seq = new unsigned char[reflength]();

	strcpy( (char*) Ref_seq, ref.c_str());*/

	fgets(Comp_seq,max,Cstream);

	/*if(Comp_seq[0] == '0'){

		fgets(Comp_seq,max,Cstream);

	}*/

	if(Comp_seq[0] == '1'){//lowcase letters compress method

		// double de_start = 0.0;

		// double readreftime = 0.0;

		// double readreftime_start = clock();

		readReflow(Rstream,Refsequence,meta);

		// readreftime += clock() -readreftime_start;

		// readreftime = readreftime /CLOCKS_PER_SEC;

		//Tar_seq = new char[max];

		//cout << "read reference lower time : " << readreftime << endl;

		double decodetime = clock();

			fgets(Comp_seq,max,Cstream);

			string metade;

			if(Comp_seq != 0){

				metade =Comp_seq;//.c_str();

			}else{

				metade = meta.at(0);

			}

			/*ref = Refsequence.at(0);

			int reflength = Refsequence.at(0).length();

			unsigned char *Ref_seq = new unsigned char[reflength]();

			strcpy( (char*) Ref_seq, ref.c_str());*/

			while(fgets(Comp_seq,max,Cstream)!= NULL){

				if(Comp_seq[0] == eolChar1) {

					continue;

				}

				// if(tar_len > 205335000){

				// 	break;

				// }

				if(Comp_seq[0] <0x40){

					temp = Comp_seq;

					//flag_mis = false;

					while(*temp != eolChar1){

						if(*temp == eolChar1){

							++ temp;

							continue;

						}

						// if(tar_len > 909600){

						// 	int mm = 0;

						// }

						if(*temp < 0x40){

							if( Comp_seq[0] == ' '&& Comp_seq[1] == ' '){//flag_mis &&

								curpos = mis_len;

							}else{

								if(*temp == 0x20){

									sscanf(temp, "%u", &curpos);

									curpos = 0 - curpos;

								}else{

									sscanf(temp, "%u", &curpos);

									//curpos = refpos + curpos;

								}

								++temp;

								while(*temp != eolChar1 && *temp != eolChar2 && *temp !=0x20 && *temp < 0x40){

									++temp;

								}

							}

							// if(curpos == 30526555){

							// 	int www = 0;

							// }

							refpos +=curpos;

							totlen = 0;

						//	flag_mis = false;

							while(*temp != eolChar1 && *temp != eolChar2 && *temp < 0x40 && *temp){

								sscanf(temp,"%u", &samelen);

								totlen += samelen;

								for(int a = 0; a < samelen; a++){

									Tar_seq[tar_len] = Ref_seq[refpos];

									tar_len ++;

									refpos ++;

								}

								while(*temp == ' '){

									++ temp;

								}

							//	flag_mis = false;

								mis_len = 0;

								while(*temp != eolChar1 && *temp != eolChar2 && *temp != 0x20 && *temp < 0x40){

									++ temp;

								}

								if(*temp != eolChar1 && *temp < 0x40 && *temp){

									sscanf(temp, "%u", &diflen);

									totlen += diflen;

									for(int b = 0; b < diflen; b ++){

										Tar_seq[tar_len] = Ref_seq[refpos] ^ (char)0x20;

										tar_len++;

										refpos ++;

									}

									while(*temp ==' '){

										++ temp;

									}

									while(*temp != eolChar1 && *temp != eolChar2 && *temp != ' ' && *temp < 0x40){

										++ temp;

									}

								}

								// if(tar_len > 9600){

								// 	int mm =0;

								// }

							}

							prepos = refpos;

						}else{

							while(*temp > 0x40 && *temp != eolChar1){

								Tar_seq[tar_len] = *temp;

								tar_len++;

								totlen ++;

								++ temp;

								mis_len ++;

							}

							//flag_mis = true;

						}

					}

					/*if(*temp == eolChar1){

					++ temp;

					continue;

					}

					if(*temp == 0x20){

						sscanf(temp, "%u", &curpos);

						curpos = 0 - curpos;

					}else{

						sscanf(temp, "%u", &curpos);

						//curpos = refpos + curpos;

					}

					++temp;

					while(*temp != eolChar1 && *temp != eolChar2 && *temp !=0x20){

						++temp;

					}

					//++temp;

					refpos +=curpos;

					totlen = 0;

					while(*temp != eolChar1 && *temp != eolChar2 && *temp ){

						sscanf(temp, "%u", &samelen);

						totlen += samelen;

						for(int a =0;a < samelen;a++){

							Tar_seq[tar_len] = Ref_seq[refpos];

							tar_len ++;

							refpos ++;

						}

						++ temp;

						while(*temp != eolChar1 && *temp != eolChar2 && *temp !=0x20){

							++ temp;

						}

						if(*temp != eolChar1 && *temp){

							sscanf(temp, "%u", &diflen);

							totlen += diflen;

							for(int b = 0;b < diflen;b++){

								Tar_seq[tar_len] = Ref_seq[refpos] ^ (char)0x20;;

								tar_len++;

								refpos ++;

							}

							++ temp;

							while(*temp != eolChar1 && *temp != eolChar2 && *temp !=' '){

								++ temp;

							}

						}

						

					}

					prepos = refpos;*/

				}else{

					temp = Comp_seq;

					if((Comp_seq[0] == 'N' || Comp_seq[0] == 'n') && Comp_seq[1] <0x40){

						char N = *temp;

						int Nlen;

						++ temp;

						if(Comp_seq[1] > 0xA){

							sscanf(temp,"%u",&Nlen);

							for(int c = 0;c < Nlen; c++){

								Tar_seq[tar_len] = N;

								tar_len++;

							}

							refpos += Nlen;

							mis_len = 0;

							while(*temp != eolChar1 && *temp !=eolChar2

							      && *temp != ' ' && *temp < 0x40){

									  ++ temp;

							}

							//flag_mis = true;

						}else{

							//char N = *temp;

							Tar_seq[tar_len] = N;

							tar_len ++;

							mis_len ++;

							//flag_mis = true;

							while(*temp != eolChar1 && *temp !=eolChar2

							      && *temp != ' ' && *temp < 0x40){

								++temp;

							}

						}

					}else{

						while(*temp > 0x40 && *temp != eolChar1){

							Tar_seq[tar_len] = *temp;

							tar_len ++;

							totlen ++;

							++temp;

							mis_len ++;

						}

						//flag_mis = true;

					}

				}

				

				

			}

			decodetime = clock() - decodetime;

			decodetime = decodetime / CLOCKS_PER_SEC;

			cout << "decodetime : " << decodetime << endl;

			int mm = 0;

	}else if(Comp_seq[0]== '0'){//upcase letters compress method

		

		readRef(Rstream,Refsequence,meta);

		

		fgets(Comp_seq,max,Cstream);

		string metade;

			//Tar_seq = new char[max];

		if(Comp_seq[0] != '0'){

			metade =Comp_seq;//.c_str();

		}else{

			metade = meta.at(0);

		}

		fgets(Comp_seq,max,Cstream);

				temp = Comp_seq;

				int begin,length;

				sscanf(temp, "%u",&pos_vec_len);

				int n = 0;

				for( n = 0; n < pos_vec_len;n ++){

					while(*temp != eolChar1 && *temp != eolChar2 && *temp !=0x20 && *temp < 0x40){

						++temp;

					}

					sscanf(temp,"%u",&begin);

					pos_vec[n].begin = begin;

					temp ++;

					while(*temp != eolChar1 && *temp != eolChar2 && *temp !=0x20 && *temp < 0x40){

						++temp;

					}

					sscanf(temp,"%u",&length);

					pos_vec[n].length = length;

					temp ++;

					//n++;

				}

   				while(fgets(Comp_seq,max,Cstream)!= NULL){

					if(Comp_seq[0] == eolChar1) {

						continue;

					}

					// if(tar_len ==  1509770){

					// 	cout<< "error is" <<endl;

					// }

					int m = 0;

					temp = Comp_seq;

					//flag_mis = false;

					if(Comp_seq[0] <0x40){

						temp = Comp_seq;

						while(*temp != eolChar1){

							if(*temp == eolChar1){

								++ temp;

								continue;

							}

							if(*temp < 0x40){

								if(/*flag_mis &&*/ Comp_seq[0] == ' '&& Comp_seq[1] == ' '){

									curpos = mis_len;

								}else{

									if(*temp == 0x20){

										sscanf(temp,"%u",&curpos);

										curpos = 0 - curpos;

									}else{

										sscanf(temp,"%u",&curpos);

									}

									++temp;

									while(*temp != eolChar1 && *temp != eolChar2 && *temp !=0x20 && *temp < 0x40){

										++temp;

									}

									flag_mis = false;

								}

								

								refpos += curpos;

								totlen = 0;

								mis_len = 0;

								curpos = 0;

								while(*temp != eolChar1 && *temp != eolChar2 && *temp < 0x40 && *temp){

									sscanf(temp,"%u",&uplen);

									totlen += uplen;

									for(int a = 0; a < uplen; a++){

  										char ch = Ref_seq[refpos];//Ref_seq

										if(isupper(ch)){

											Tar_seq[tar_len] = ch;

											tar_len ++;

											refpos ++;

										}else{

											Tar_seq[tar_len] = toupper(ch);

											tar_len ++;

											refpos ++;

										}

										// if(tar_len ==  1509770){

										// 	cout<< "error is" <<endl;

										// }

									}

									// if(tar_len ==  1509770){

									// 	cout<< "error is" <<endl;

									// }	

									//++ temp;

									while(*temp == ' '){

										++ temp;

									}

									while(*temp != eolChar1 && *temp != eolChar2 

									 && *temp != ' ' && *temp < 0x40){

										++ temp;

									}

									prepos = refpos;

								}

							}else{

								while(*temp >0x40 && *temp != eolChar1){

									Tar_seq[tar_len] = *temp;

									tar_len++;

									totlen ++;

									++ temp;

									mis_len ++;

								}

								flag_mis =true;

							}

						}

					}else{

						temp = Comp_seq;

						if(Comp_seq[0] =='N' && Comp_seq[1] < 0x40 ){

							char N = *temp;

							int Nlen;

							++ temp;

							if(Comp_seq[1] > 0xA){

								sscanf(temp,"%u",&Nlen);

								for(int b = 0;b < Nlen; b ++){

									Tar_seq[tar_len] = 'N';

									tar_len ++;

								}

								while(*temp != eolChar1 && *temp != eolChar2 

									&& *temp != ' ' && *temp < 0x40){

									++ temp;

								}

								refpos += Nlen;

								flag_mis = true;

								mis_len = 0;

							}else{

								Tar_seq[tar_len] = 'N';

								tar_len ++;

								mis_len ++;

								flag_mis = true;

								while(*temp != eolChar1 && *temp != eolChar2 

									&& *temp != ' ' && *temp < 0x40){

									++ temp;

								}

							}

							// if(tar_len ==  1509770){

							// 	cout<< "error is" <<endl;

							// }

						}else{

							while(*temp > 0x40 && *temp != eolChar1){

								Tar_seq[tar_len] = *temp;

								tar_len++;

								totlen ++;

								++ temp;

								mis_len ++;

							}

							flag_mis = true;

						}

 						// if(tar_len ==  1509770){

						// 	cout<< "error is" <<endl;

						// }

					}

				}

			int k = 0;	

			int o = 0;

			for(int x = 0;x < pos_vec_len ;x ++){

				k += pos_vec[x].begin;

				o += pos_vec[x].begin;;

				

				int l = pos_vec[x].length;

				o += pos_vec[x].length;

				if(x == pos_vec_len){

					int ww = 0;

				}

				/*if(o>242818020){

						int ww= 0;

					}*/

				for(int j = 0;j < l ; j ++){

					Tar_seq[k] = tolower(Tar_seq[k]);

					k++;

					/*if(o>242818020){

						int ww= 0;

					}*/

				}

				/*if(o>242818020){

						int ww= 0;

				}

				int ww = 0;*/

			}

			//int ww = 9;

		

		}	

		

}



int main(int argc, char **argv){

	int c;

	int rmq =0;

	bool flag = false;;

	while ((c = getopt (argc, argv, "r:")) != -1)

		switch (c){

		case 'r':

	        rmq = 1;

        	break;

		if (isprint (optopt))

                	fprintf (stderr, "Unknown option `-%c'.\n", optopt);

             	else

               		fprintf (stderr,"Unknown option character `\\x%x'.\n", optopt);

		return 1;

		default:

		abort ();

	}

	char defile[150];

	double decompresstime = 0.0;

	double startTime = clock();

	sprintf(defile,"%s",argv[argc-4]);

	string dfile = defile;

	//int ret = mkdir(dfile.c_str(), 0777);

	LMSRGDE::bsc::BSC_decompress(dfile.c_str(), (dfile + ".tar").c_str());

	//string tarfnd = generateString("LCPDE",5);

	//ret = mkdir(tarfnd.c_str(), MODE);

	char tempdf[150];

	//string tdf = tempdf;

	sprintf(tempdf,"%s",argv[argc-3]);

	string tdf = tempdf;

	//string tempcf = tempdf;

	string tarcmd = "tar -xf " + (dfile + ".tar") + " -C " + tdf;

	system(tarcmd.c_str());

   vector<string> filename = {"chr1.fa", "chr2.fa", "chr3.fa", "chr4.fa", //default chr name list

                    "chr5.fa", "chr6.fa", "chr7.fa", "chr8.fa", "chr9.fa", "chr10.fa", 

                    "chr11.fa", "chr12.fa", "chr13.fa", "chr14.fa", "chr15.fa", "chr16.fa", "chr17.fa", 

                    "chr18.fa", "chr19.fa", "chr20.fa", "chr21.fa", "chr22.fa", "chrX.fa", "chrY.fa"};

   for(int d = 0 ; d < 24; d ++){

	   char ref[150];char tar[150];char code[150];char decode[150];

		FILE *Rstream;

		FILE *Cstream;

		FILE *Gstream;

		FILE *Tstream;

		vector<string> taxa;

		vector<string> sequences;

		int leng = 1UL<<22;

		if (argc < 3) {

			fprintf(stderr, "usage: %s file.fasta k\n", argv[0]);

			return EXIT_FAILURE;

		}

		sprintf(ref,"%s%s",argv[argc-5],filename[d].c_str());

		sprintf(code,"%s%s",argv[argc-3],filename[d].c_str());

		//sprintf(code,"%s%s",filetest,filename[d].c_str());

		sprintf(tar,"%s%s",argv[argc-1],filename[d].c_str());

		sprintf(decode,"%s%s",argv[argc-2],filename[d].c_str());

		Rstream = fopen(ref, "r");

		Cstream = fopen(code, "r");

		Gstream = fopen(decode, "w");

		Tstream = fopen(tar, "r");

		if (!Rstream) {

			perror("can't open input file");

			fprintf(stderr, "usage: %s file.fasta k\n", argv[0]);

			return EXIT_FAILURE;

		}

		if (!Cstream) {

			perror("can't open compress file");

			fprintf(stderr, "usage: %s file.fasta k\n", argv[0]);

			return EXIT_FAILURE;

		}

		if (!Tstream) {

			perror("can't open compress file");

			fprintf(stderr, "usage: %s file.fasta k\n", argv[0]);

			return EXIT_FAILURE;

		}

		//int k = atoi(argv[argc-1]);

		//vector<string> Refsequence;

		double time=0.0;

		double start =clock();

		cout << "decompressing : " << filename[d] << endl;

		vector<string> meta;

		// fseek(Rstream,0,SEEK_END);

		// int length = ftell(Rstream);

		// fseek(Rstream,0,SEEK_SET);

		//Ref_seq = new char[length];

		//readRef(Rstream,Refsequence);

		int max_seq = 1 << 30;

		Comp_seq = new char[max_seq];

		temp= new char[max_seq];

		Ref_seq = new char[max_seq];

		Tar_seq = new char[max_seq];

		decodeCompress(Gstream,Cstream,Rstream,meta);

		flag = false;

		//flag = true;

		if(flag){

			Ver_seq = new char[max_seq];

			readTar(Tstream);

			Verification(Ver_seq,Tar_seq);

		}

		

		outputdecode(Gstream,Tar_seq,decode,meta);

		

   }

	cout << "done" << endl;

	decompresstime += clock() - startTime;

	decompresstime = decompresstime / CLOCKS_PER_SEC;

	cout << "decompression time : " << decompresstime << endl;

	return EXIT_SUCCESS;

}

