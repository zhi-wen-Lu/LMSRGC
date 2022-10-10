//#include "parser.h"
#include <malloc.h>
#include <time.h>
//#include "kmacs.h"
#include <chrono>
#include <thread>
#include <iostream>
#include <vector>
#include <fstream>
#include "kvec.h"
#include <algorithm>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include "sais.h"
#include <sstream>
#include "libbsc/bsc.h"
#include "include/gpu_lcp.h"
#include "suffix_array.h"
#include <cstdint>

using namespace std;
const int min_size = 1<<23;
int pos_vec_len;
struct POSITION_RANGE{
	int begin, length;
};

typedef struct match_t{
	size_t len;
	size_t refidx;
	size_t taridx;
	match_t(){}
	match_t(size_t _refidx,size_t _taridx,size_t _len): refidx(_refidx),taridx(_taridx),len(_len){}
}match_t;

typedef struct { size_t n,m; match_t *a;}match_v;

//char* Tar_seq;
int Tar_len;
int Ref_len;
bool compareidx (match_t &a,match_t &b)
{
	return a.taridx < b.taridx;
}

void GetLcpFromMemory(unsigned char* input_string, uint32_t* sa, int* lcp, int n, int thread_cnt = 72)
{
	/*auto before_time = std::chrono::steady_clock::now();

	//ÉêÇërankÄÚ´æ£¬¼ÆËãrankÊý×é
	cout << "calc rank..." << endl;*/
	int* rank = new int[n];
	std::vector<std::thread> threads;
	const int step = n / thread_cnt;  //Ã¿¸öÏß³Ì¼ÆËãÊýÁ¿
	for (int i = 0; i < thread_cnt; i++)
	{
		int start_idx = i * step;
		int end_idx;
		if (i == thread_cnt - 1)  end_idx = n;
		else end_idx = start_idx + step;
		threads.emplace_back(std::thread([start_idx, end_idx, &sa, &rank] {
			for (int i = start_idx; i < end_idx; i++)  rank[sa[i]] = i;
			}));
	}
	for (int i = 0; i < thread_cnt; i++) threads[i].join();
	auto rank_time = std::chrono::steady_clock::now();
	threads.clear();
	for (int i = 0; i < thread_cnt; i++)
	{
		//auto thread_time = std::chrono::steady_clock::now();
		int start_idx = i * step;
		int end_idx;
		if (i == thread_cnt - 1)  end_idx = n;
		else end_idx = start_idx + step;
		threads.emplace_back(std::thread([start_idx, end_idx, &input_string, &sa, &lcp, &rank] {

			int i, k = 0;
			for (i = start_idx; i < end_idx; i++)
			{
				if (k) k--;
				if (rank[i] == 0) continue;
				int j = sa[rank[i] - 1];
				while (input_string[i + k] == input_string[j + k]) k++;
				lcp[rank[i]] = k;			}
			}));
			/*auto thread_end_time = std::chrono::steady_clock::now();
			double d3 = std::chrono::duration<double>(thread_end_time - thread_time).count();
			std::cout << "single thread time : "<< d3<<endl;*/
	}
	for (int i = 0; i < thread_cnt; i++) threads[i].join();
	auto after_time = std::chrono::steady_clock::now();
	//double d1 = std::chrono::duration<double>(rank_time - before_time).count();
	double d2 = std::chrono::duration<double>(after_time - rank_time).count();
	//std::cout << "rank use time:" << d1 << "Ãë" << std::endl;
	std::cout << "lcp use time:" << d2 << "Ãë" << std::endl;
	threads.clear();
	/*FILE *rp = fopen("/itmslhppc/itmsl0105/workspace/result/test-lcp/RK.txt","w+");
	fwrite(rank,sizeof(int),n,rp);
	fclose(rp);*/
	delete[] rank;
}

bool comparebylen (match_t &a,match_t &b)
{
	return a.len > b.len;
	//return 0;
}

bool comparebylenidx (match_t &a,match_t &b)
{
	if(a.len > b.len){
		return a.len > b.len;
	}else if(a.len == b.len){
		return a.taridx < b.taridx;
	}
	//return a.len > b.len;
	return 0;
}

//  POSITION_RANGE *pos_vec= new POSITION_RANGE[min_size];
//  POSITION_RANGE *n_vec= new POSITION_RANGE[min_size];

void sortbylen(int tmpz,int tmptaridx,int tmprefidx,int tmplen,int tmpall,match_v *ml){
	int w= 0;
	for(w = tmpz ; w < tmpall;w ++){
		if(tmplen < ml->a[w + 1].len){
			ml->a[w].taridx = ml->a[w+1].taridx;
			ml->a[w].refidx = ml->a[w+1].refidx;
			ml->a[w].len = ml->a[w+1].len;			
		}else{
			// int tmpt = tmptaridx;
			// int tmpr = tmprefidx;
			// int tmpl = tmplen;
			ml->a[w].taridx = ml->a[w + 1].taridx;
			ml->a[w].refidx = ml->a[w + 1].refidx;
			ml->a[w].len = ml->a[w + 1].len;
			ml->a[w + 1].taridx = tmptaridx;
			ml->a[w + 1].refidx = tmprefidx;
			ml->a[w + 1].len = tmplen;
			break;
		}
			
	}
	return ;
}

void sortbymld(int tmpz,int tmpall,int tmplen,match_v *ml, int *mld){
	int u = 0;
	int v = 0;
	v = mld[tmpz];
	for(u = tmpz ; u < tmpall; u ++){
		if(tmplen < ml->a[mld[u + 1]].len){
			//v = tmpz;
			mld[u] = mld[u +1];
		}else{
			mld[u] = mld[u + 1];
			mld[u + 1] = v;
			break;
		}
	}
}

void readTar(FILE *stream, vector<string>&Tarsequence,vector<string>&taxatar, POSITION_RANGE *pos_vec,POSITION_RANGE *n_vec){
	vector<char> seqBuffer;
	vector<char> taxaBuffer;
	char c=getc(stream);
	bool flag = true;
	bool flag_n = false;
	int n_vec_len = 0;
	int n_len = 0;
	int c_len = 0;
	int all_len = 0;;
	pos_vec_len = 0;
	int letters_len = 0;
	while(c!=EOF){
		if(c=='>'|| c =='@'){
			taxaBuffer.push_back(c);
			while((c=getc(stream))!='\n'){
				taxaBuffer.push_back(c);		
			}
		}
		else{
			while((c=getc(stream))!='>' && c!=EOF){
				if(c == '\n'){
					continue;
				}
				if(islower(c)){
					if(flag){
						flag = false;
						pos_vec[pos_vec_len].begin = letters_len;
						letters_len = 0;
					}
					c=toupper(c);
				}else{
					if(isupper(c)){
						if(!flag){
							flag = true;
							pos_vec[pos_vec_len].length = letters_len;
							pos_vec_len++;
							letters_len = 0;
						}
					}

				}
				if(c == 'N'){
					if(!flag_n){
						n_vec[n_vec_len].begin = all_len;
						//n_vec_len++;	
					}
					n_len ++;
					flag_n = true;
				}else{
					if(flag_n){
						n_vec[n_vec_len].length = n_len;
						n_vec_len++;
						n_len = 0;
						flag_n = false;
					}
				}
				//c=toupper(c);
				letters_len ++;
				all_len ++;
				/*if(c != 'N'){
					seqBuffer.push_back(c);	
					c_len ++;
				}*/
				seqBuffer.push_back(c);	
				c_len ++;
				
			}	
		}
		if(flag_n){
			n_vec[n_vec_len].length = n_len;
			n_vec_len++;
		}
		if(!flag){
			//int ww = 0;
			//flag = true;
			pos_vec[pos_vec_len].length = letters_len;
			pos_vec_len++;
			letters_len = 0;
		}
		if(seqBuffer.size()>0){
			string taxaString(taxaBuffer.begin(),taxaBuffer.end());
			string seqString(seqBuffer.begin(),seqBuffer.end());
			taxatar.push_back(taxaString);
			Tarsequence.push_back(seqString);
			seqBuffer.clear();
			taxaBuffer.clear();
		}
	}
	// *tar_len = all_len;
	return ;
}

void readRef(FILE *stream, vector<string>&Refsequence,vector<string>&taxaref){
	vector<char> seqBuffer;
	vector<char> taxaBuffer;
	int n_len = 0;
	char c=getc(stream);
	while(c!=EOF){
		if(c=='>' || c == '@'){
			taxaBuffer.push_back(c);
			while((c=getc(stream))!='\n'){
				taxaBuffer.push_back(c);		
			}
		}
		else{
			while((c=getc(stream))!='>' && c!=EOF){
				if(!isalpha(c)){
					continue;
				}
				c=toupper(c);
				//seqBuffer.push_back(c);	
				if(c != 'N'){
					seqBuffer.push_back(c);	
				}else{
					char ch;
					ch = '1';
					seqBuffer.push_back(ch);
				}
				
			}	
		}
		if(seqBuffer.size()>0){
			int reflen = seqBuffer.size();
			while (reflen >= 0)
			{
				char ch;
				ch = seqBuffer[reflen];
				if(ch =='A'||ch =='C'||ch =='G'||ch =='T'){
					if(ch =='A') ch = 'T';
					else if(ch =='C') ch = 'G';
					else if(ch == 'G') ch = 'C';
					else if(ch == 'T') ch = 'A';
					seqBuffer.push_back(ch);
				}
				//seqBuffer.push_back(ch);
				reflen --;
			}
			string taxaString(taxaBuffer.begin(),taxaBuffer.end());
			//string seqString(seqBuffer.begin(),seqBuffer.end());
			string seqString(seqBuffer.data(),seqBuffer.size());
			taxaref.push_back(taxaString);
			Refsequence.push_back(seqString);
			seqBuffer.clear();
			taxaBuffer.clear();
		}
	}
	return ;


}

void ReadOriTar(FILE *Tstream,vector<string>&tarerence){
	vector<char> seqBuffer;
	vector<char> taxaBuffer;
	int n_len = 0;
	char c=getc(Tstream);
	while(c!=EOF){
		if(c=='>'|| c =='@'){
			while((c=getc(Tstream))!='\n'){
				taxaBuffer.push_back(c);		
			}
		}
		else{
			while((c=getc(Tstream))!='>' && c!=EOF){
				if(!isalpha(c)){
					continue;
				}
				//c=toupper(c);
				
				seqBuffer.push_back(c);	
			}	
		}
		if(seqBuffer.size()>0){
			//string taxaString(taxaBuffer.begin(),taxaBuffer.end());
			string seqString(seqBuffer.begin(),seqBuffer.end());
			//taxa.push_back(taxaString);
			tarerence.push_back(seqString);
			seqBuffer.clear();
			taxaBuffer.clear();
		}
	}
	return ;
}

void ReadOriRef(FILE *Rstream, vector<string>&Reference){
	vector<char> seqBuffer;
	vector<char> taxaBuffer;
	int n_len = 0;
	char c=getc(Rstream);
	while(c!=EOF){
		if(c=='>'|| c =='@'){
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
				seqBuffer.push_back(c);				
			}	
		}
		if(seqBuffer.size()>0){
			int reflowlen = seqBuffer.size();
			while (reflowlen >= 0)
			{
				char ch;
				ch = seqBuffer[reflowlen];
				if(ch =='A'||ch =='C'||ch =='G'||ch =='T'||ch =='a'||ch =='c'||ch =='g'||ch =='t'){
					if(ch == 'a') ch = 't';
					else if(ch =='A') ch = 'T';
					else if(ch =='C') ch = 'G';
					else if(ch =='c') ch = 'g';
					else if(ch =='G') ch = 'C';
					else if(ch =='g') ch = 'c';
					else if(ch =='T') ch = 'A';
					else if(ch =='t') ch = 'a';
					seqBuffer.push_back(ch);
				}
				reflowlen --;
			}
			
			//string taxaString(taxaBuffer.begin(),taxaBuffer.end());
			string seqString(seqBuffer.begin(),seqBuffer.end());
			//taxa.push_back(taxaString);
			Reference.push_back(seqString);
			seqBuffer.clear();
			taxaBuffer.clear();
		}
	}
	return ;
}

void saveDATA(int *sign,FILE *Tstream,FILE *Rstream,FILE *Ostream,vector<string>&Tarsequence,vector<string>Refsequence, match_v *ml, vector<string> taxaref,vector<string> taxatar,string &upBuffer,char* out,POSITION_RANGE *pos_vec,POSITION_RANGE *n_vec){

		string meta_data_ref = taxaref.at(0);
		//int meta_length_ref = taxaref.at(0).length();
		string meta_data_tar = taxatar.at(0);
		//int meta_length_tar = taxaref.at(0).length();
		string meta;
		if(strcmp(meta_data_ref.c_str(),meta_data_tar.c_str()) == 0){//std::get<0>(taxaref),std::get<0>(taxatar)
			meta.append("0\n");
		}else{
			meta.append(meta_data_tar+ "\n");
		}
	upBuffer.append(std::to_string(pos_vec_len) + " ");
	for(int a = 0; a < pos_vec_len; a ++){
		//fprintf(fp, "%d %d",pos_vec[a].begin,pos_vec[a].length);
		upBuffer.append(std::to_string(pos_vec[a].begin)+ " "+std::to_string(pos_vec[a].length)+" " );
	}
	upBuffer.append("\n");
	delete [] pos_vec;
	string tar,ref;
	char ch;
	int refidx,len;
	bool flag = false;
	tar =  Tarsequence.at(0);
	ref = Refsequence.at(0);


	int tarlength = Tarsequence.at(0).length();
	int reflength = Refsequence.at(0).length();
	unsigned char *Tar = new unsigned char[tarlength]();
	strcpy( (char*) Tar, tar.c_str());
	unsigned char *Ref = new unsigned char[reflength]();
	strcpy( (char*) Ref, ref.c_str());
	
	ref.clear();
	//ref.shrink_to_fit();
	//string().swap(ref);
	tar.clear();
	string code;
	//code.reserve((1UL<<22));
	// string chcode;
	// chcode.reserve((1UL<<22));
	int  c = 0,refend = 0,m = 0,mis_len = 0;flag = false;
	bool flag_mis = false;int taridx;bool flag_N = false;
	int _refidx,_taridx,_len,_tarend,_mis_len,_m,_refend,mum,match_len;
	for( m = m ;m < tarlength;m++){		
		if(sign[m] != -1){
			int n = sign[m];
			match_t temp = ml->a[n];
			refidx = temp.refidx;
			taridx = temp.taridx;
			len = temp.len;
			if(m > 900211){
 				int ww = 0;
			}			
 			if(Tar[taridx] == 'N'){
				code.append("\n");
				code.append("N" + std::to_string(len));
				refend += len;
				m += len - 1;
				flag_N = true;
				mis_len = 0;
				continue;
			}				
			if(abs(refidx - refend) < 30000){
				//code.append("\n");
				flag_mis = true;
				flag_N = false;
				if(mis_len != refidx - refend){
					code.append("\n");
					
					if((refidx - refend) < 0){
						code.append(" ");
					}
					code.append(std::to_string(abs(refidx - refend)) + " " + std::to_string(len));
					mis_len = 0;
					
				}else{
					code.append("\n""  " + std::to_string(len));
					mis_len = 0;
				}
				m += (len - 1);
				refend = refidx + len;
			}else{		
				_m = m;
				_refidx = refend + mis_len;
				_refend = refend + mis_len;
				_tarend = taridx + len + mis_len;
				int match_len = 0;
				_mis_len = 0;								
					while(_refend < reflength && Tar[_m] != Ref[_refend] 
						&& _m < tarlength -3){
							_mis_len ++;
							_refend ++;
							_m ++;
						}
					while(_refend < reflength && _m < tarlength -3 &&Tar[_m] == Ref[_refend]
						&& sign[_m] == n){
						_m ++;
						match_len ++;
						_refend ++;
					}
					if(match_len > len *0.9){
						if(_mis_len != 0){
							if(flag_mis){
								code.append("\n");
								//chcode.append("\n");
							}
							for(int c = 0;c < _mis_len ; c++){
								char ch = Tar[m];
								string mis;
								mis = ch;
								code.append(mis);
								//chcode.append(mis);
								m ++;
							}
							code.append("\n""  " + std::to_string(match_len));
							flag_mis = true;
							m += match_len;
							mis_len = 0;
							refend = _refend;
						}else{
							code.append("\n""  " + std::to_string(match_len));
							flag_mis = true;
							m += match_len;
							mis_len = 0;
							refend = _refend;
						}
					}else{
						code.append("\n");
						if(refidx - refend < 0){
							code.append(" ");
						}
						code.append(std::to_string(abs(refidx - refend)) + " "+ std::to_string(len));
						mis_len = 0;
						m += len;
						flag_mis = true;
						refend = refidx + len;
					}
					while(sign[m]  == n){
						if(flag_mis){
							code.append("\n");
							//chcode.append("\n");
						}
						char ch = Tar[m];
						string mis;
						mis = ch;
						m ++;
						code.append(mis);
						//chcode.append(mis);
						flag_mis = false;
						mis_len ++;
					}
					m --;
			}
		}else if(mis_len < 21){ 
			_m = m;
			_refidx = refidx;
			_refend = refend + mis_len;
			_tarend = taridx + len + mis_len;
			_mis_len = 0;
			int match_len = 0;
			//int mum = sign[m - 1];
			if(flag_mis || flag_N){
				code.append("\n");
				//chcode.append("\n");
				mum = sign[m -1];
				flag_mis = false;
				flag_N = false;
			}
			while(_mis_len <= 2 && _refend < reflength && Tar[_m] != Ref[_refend] 
			  && _m < tarlength - 3 && sign[_m] == -1 ){
				_mis_len ++;
				_refend ++;
				_m ++;
			}
			if(_mis_len == 0){
				char ch = Tar[m];
				string mis;
				mis = ch;
				code.append(mis);
				//chcode.append(mis);
				mis_len ++;
			}else{
				match_len = 0;
				while( _refend < reflength &&sign[_m] == -1 && Tar[_m] == Ref[_refend] 
				       && _m < tarlength ){
					match_len ++;
					_m ++;
					_refend ++;
				}
				if(match_len >= 3){
					for(int c = 0; c < _mis_len; c++){
						char ch = Tar[m];
						string mis;
						mis = ch;
						code.append(mis);
						//chcode.append(mis);
						m ++;
					}
					code.append("\n""  " + std::to_string(match_len));
					//code.append("\n");
					flag_mis = true;
					m += (match_len - 1);
					//m -=1;
					mis_len += (match_len + _mis_len);
					refend += mis_len;
					mis_len = 0;
				}else{
					char ch = Tar[m];
					string mis;
					mis = ch;
					code.append(mis);
					//chcode.append(mis);
					mis_len ++;

				}
			}
		}else{
			char ch = Tar[m];
			string mis;
			mis = ch;
			code.append(mis);
			//chcode.append(mis);
			mis_len ++;
		}
		//code.append("\n");
	}
	 
	// delete n_vec;
	 delete sign;
	//free(ml);
	//delete [] ml;
	delete [] Tar;
	delete [] Ref;
	code.append("\n");
	vector<string> tarerence;
	fseek(Tstream,0,SEEK_SET);
	ReadOriTar(Tstream,tarerence);
	string tarcode = tarerence.at(0);
	int Tar_len = tarerence.at(0).length();
	unsigned char *Tar_seq = new unsigned char[Tar_len]();
	strcpy((char*) Tar_seq, tarcode.c_str());
	vector<string> Reference;
	fseek(Rstream,0,SEEK_SET);	
	ReadOriRef(Rstream,Reference);
	string refcode;
	refcode = Reference.at(0);
	int Ref_len = Reference.at(0).length();
	unsigned char *Ref_seq = new unsigned char[Ref_len]();
	strcpy((char*) Ref_seq, refcode.c_str());
	tarerence.clear();
	Reference.clear();
	refcode.clear();
	tarcode.clear();
	string lowcode;
	lowcode.reserve((1UL<<22));
	 m = 0;int i = 0;
	refend = 0;mis_len = 0;
	 //int tarend;
	bool flag_poi = false;
	flag_mis = false;
	flag_N = false;
	mum = -1;
	int mum_match_len = 0;int x = 0;
	int index = 0;

	int ref_idx = 0;int matchlen = 0;int tarend = 0;int a = 0;int offset= 0;
	while(x < code.size()){
		char digit_string[10] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
		lowcode.append("\n");
		
		if(isdigit(code[x])){
			//lowcode.append("\n");
			while(isdigit(code[x])){
				digit_string[index] = code[x];
				x++;
				index ++;
			}
			offset = atoi(digit_string);
			if(offset == 30526555){
				int ww= 0;
			}
			lowcode.append(std::to_string(offset));
			ref_idx += offset;
			x ++;
			index = 0;
			while (code[x] == ' ')
			{
				x ++;
			}
			
			char digit_string[10] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
			while(isdigit(code[x])){
				digit_string[index] = code[x];
				x ++;
				index ++;
			}
			index = 0;
			matchlen = atoi(digit_string);
			tarend += (matchlen + mis_len);
			mis_len = 0;
			//int slen = 0;
			while(a < tarend){
				int slen = 0;
				while(a < tarend && Tar_seq[a] == Ref_seq[ref_idx] ){
					slen ++ ;
					a ++;
					ref_idx ++;
				}
				lowcode.append(" " + std::to_string(slen));
				slen = 0;
				while(a < tarend && Tar_seq[a] != Ref_seq[ref_idx]){
					slen ++;
					a ++;
					ref_idx ++;
				}
				if(slen > 0){
					lowcode.append(" " + std::to_string(slen));
				}
			}
		}else if(code[x] == ' ' && isdigit(code[x+1])){
			//lowcode.append("\n");
			while(code[x] == ' '){
				x ++;
			}
			char digit_string1[10] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
			index = 0;
			while(isdigit(code[x])){
				//x ++;
				digit_string1[index] = code[x];
				x ++;
				index ++;
			}
			offset = atoi(digit_string1);
			ref_idx += (0 - offset);
			while(code[x] == ' '){
				x ++;
			}
			//char digit_string[10] = {-1};
			lowcode.append(' ' + std::to_string(offset));
			index = 0;
			while(isdigit(code[x])){
				digit_string[index] = code[x];
				x ++;
				index ++;
			}
			index = 0;
			matchlen = atoi(digit_string);
			tarend += (mis_len + matchlen);
			mis_len = 0;
			//int slen = 0;
			while(a < tarend){
				int slen = 0;
				while(a < tarend && Tar_seq[a] == Ref_seq[ref_idx] ){
					slen ++ ;
					a ++;
					ref_idx ++;
				}
				lowcode.append(" " + std::to_string(slen));
				slen = 0;
				while(a < tarend && Tar_seq[a] != Ref_seq[ref_idx]){
					slen ++;
					a ++;
					ref_idx ++;
				}
				if(slen > 0){
					lowcode.append(" " + std::to_string(slen));
				}
			}
		}else if(code[x] == ' ' && code[x+1] == ' '){
			while(code[x] == ' '){
				x ++;
			}
			ref_idx += mis_len;
			lowcode.append(" ");
			while (isdigit(code[x]))
			{
				digit_string[index] = code[x];
				x ++;
				index ++;
			}
			index = 0;
			matchlen = atoi(digit_string);
			if(matchlen == 9211){
				int mmm = 0;
			}
			tarend += (matchlen + mis_len);
			mis_len = 0;
			//int slen = 0;
			while (a < tarend)
			{
				int slen = 0;
				while(a < tarend && Tar_seq[a] == Ref_seq[ref_idx] ){
					slen ++ ;
					a ++;
					ref_idx ++;
				}
				lowcode.append(" " + std::to_string(slen));
				slen = 0;
				while(a < tarend && Tar_seq[a] != Ref_seq[ref_idx]){
					slen ++;
					a ++;
					ref_idx ++;
				}
				if(slen > 0){
					lowcode.append(" " + std::to_string(slen));
				}
			}	
		}else if(code[x] == 'N' && isdigit(code[x+1])){
			while (code[x] == 'N')
			{
				x ++;
			}
			
			while (isdigit(code[x]))
			{
				digit_string[index] = code[x];
				x ++;
				index ++;
			}
			index = 0;
			matchlen = atoi(digit_string);
			if(Tar_seq[a +1] == 'n'){
				lowcode.append('n' + std::to_string(matchlen));
			}else{
				lowcode.append('N' + std::to_string(matchlen));
			}
			
			ref_idx += matchlen;
			tarend += matchlen + mis_len;
			a += matchlen;
			mis_len = 0;
		}else if(!isdigit(code[x])){		
				while (!isdigit(code[x]) && code[x] != '\n')
				{
					if(code[x] == Tar_seq[a]){
						char ch = Tar_seq[a];
						string mis;
						mis =ch;
						lowcode.append(mis);
						x ++;
						a ++;
						//tarend ++;
						mis_len ++;
					}else if(code[x] == toupper(Tar_seq[a])){
						char ch = Tar_seq[a];
						string mis;
						mis = ch;
						lowcode.append(mis);
						x ++;
						a ++;
						mis_len ++;
						//tarend ++;
					}else{
						int error = 0;
						cout << "error"<<endl;
					}
					
				}			
			
		}
		while(code[x] == '\n'){//||code[x] == ' '
			x ++;
			//continue;
		}
	}
	delete [] Ref_seq;
	delete [] Tar_seq;
	string comstr;
	comstr.reserve((1<<14));
	 ////////////////////////////////
	 std::ofstream fn;
	 string output;
	 output = out;//Ostream.c_str(); 
	 fn.open(output);
	if(upBuffer.size() + code.size() < lowcode.size()){
		comstr.append("0\n");
		//code.append("\n");
		fn <<comstr << meta<< upBuffer<<code;
	}else {
		comstr.append("1\n");
		lowcode.append("\n");
		fn <<comstr << meta<< lowcode;
	}	
	code.clear();
	lowcode.clear();
	upBuffer.clear();
	comstr.clear();
	return ;
}
void searchLCS(int *sign,vector<string>&Tarsequence, vector<string>&Refsequence, match_v *mh, match_v *ml,POSITION_RANGE *pos_vec,POSITION_RANGE *n_vec){
	string seq;
	seq = Tarsequence.at(0) +"$"+Refsequence.at(0);
	//cout << s <<endl;
	//seq = "$"+Refsequence.at(0);
	int tarlength = Tarsequence.at(0).length();
	int reflength = Refsequence.at(0).length();
	int n = seq.length();
	unsigned char *T = new unsigned char[n+1]();
	strcpy( (char*) T, seq.c_str());
	seq.clear();
	seq.shrink_to_fit();
	uint32_t *SA = new uint32_t[n +1]();
	int *LCP = new int[n+1]();
	int taridx;
	int refidx;
	bool flag = true;
	bool flag_lcs= false;
	int matchlen;
	kv_init(*mh);
	kv_resize(match_t, *mh,tarlength);
	
	//RMQ_succinct* rmqSuc;
	double avgS1=0; double avgS2=0;

    gpuSuffixArray(/*uchar S=*/T, /*size_t n=*/n+1, /*uint32 SA=*/SA);
  
	auto LCPstart = std::chrono::steady_clock::now();
	GetLcpFromMemory(T,SA,LCP,n,30);

	auto LCPend = std::chrono::steady_clock::now();
	double LCPtime =std::chrono::duration<double>(LCPend - LCPstart).count();
	cout<< "creat LCP time : "<< LCPtime<<endl;
	
	int min=LCP[1]; 
	int minPos=1;
	int p;int tmp; int maxlen = 100;int max;int len=0;int tarend =0;
	int cnt = 30;// number threads
    std::vector<std::thread> threads;
    const int step = n / cnt;

    //double start_suff = clock();

    cout << "start....." << endl;

	auto beforeTime = std::chrono::steady_clock::now();

    for (int i = 0; i < cnt; i++)
    {
        int sx = i * step;
        int ex;

        if (i == 0) sx = 1;

        if (i == cnt - 1)  ex = n;
        else ex = sx + step;
        int number = i;

        threads.emplace_back(std::thread([sx, ex, n, &SA, &T, &LCP, &mh, tarlength, number] {
            
                int posref,posref1,posref2,nlcp,plcp,clcp,index = 0;

				//找到实际起始点
				int m = sx;
				for(; m < ex; m++) {

					//特殊情况处理
                    if(T[SA[m]] == 'N'||T[SA[m]] == '1') {
                        if(SA[m] < tarlength){
                            //ml->a[SA[m]].taridx = SA[m];
                            mh->a[SA[m]].taridx = SA[m];
                        }
                        continue;
                    }

					//位于第二个序列，下一位在第一个序列
					if (SA[m] >= tarlength && SA[m + 1] < tarlength) break;
				}
				
				//分块内，未找到起始点,直接结束
				if (m >= ex) return;

				//一直计算到分块结束
                for(; m < n; m++) {
					//特殊情况处理
                    if(T[SA[m]] == 'N'||T[SA[m]] == '1') {
                        if(SA[m] < tarlength) {
                            //ml->a[SA[m]].taridx = SA[m];
                            mh->a[SA[m]].taridx = SA[m];
                        }
                        continue;
                    }

                    if(SA[m] >= tarlength) {
						//超出当前分块，则结束
						if (m >= ex) break;

						//未超过，继续计算
                        if(SA[m + 1] < tarlength) {
                            index = m;
                            posref1 = SA[m];
                            plcp = LCP[m+1];//m + 1
                        }
                        continue;
                    }

					int z = index + 1;
					for( z= z ; z <= m; z ++){//+1
						if(LCP[z] <= plcp){
							plcp = LCP[z];
							index = z;
						}
						if(plcp == 0){
							break;
						}
						//nlcp = LCP[m];
					}
					int y = m+1;
					nlcp = LCP[y];
					
					while(SA[y] < tarlength){
						if(T[SA[y]] == 'N' || T[SA[y]] == '1'){
							nlcp = 0;
							break;  
							
						}
						if(nlcp == 0){
							break;
						}
						if(nlcp >LCP[y]){
							nlcp = LCP[y];
							//y ++;
						}
						y ++;
					}
					if(nlcp > LCP[y]){
						nlcp = LCP[y];
					}
					posref2 = SA[y];
					if(plcp >= nlcp){
						clcp = plcp;
						posref = posref1;
					}else{
						clcp = nlcp;
						posref = posref2;
					}
					if(SA[m] < tarlength && SA[m] + clcp <= tarlength){
						
						mh->a[SA[m]].taridx = SA[m];
						mh->a[SA[m]].refidx = posref;
						mh->a[SA[m]].len = clcp;
						if(SA[m] == 616){
							int ww = 0;
						}
					}

                }
               

        }));
        
    }
    	
    for (int i=threads.size()-1; i>= 0; i--)
    {
        threads[i].join();
    }

	auto afterTime = std::chrono::steady_clock::now();
    double duration_second = std::chrono::duration<double>(afterTime - beforeTime).count();
    cout << "filter SA and LCP tiem : " <<  duration_second << endl;

	int o = tarlength -5;
	while(o < tarlength){
		mh->a[o].taridx =o;
		mh->a[o].len = 1;
		mh->a[o].refidx = 0;
		o ++;
	}

	delete [] SA; delete [] LCP; delete [] T; 
	kv_init(*ml);
	kv_resize(match_t, *ml,tarlength);
	taridx = mh->a[0].taridx;
	refidx = mh->a[0].refidx;
	len = mh->a[0].len;
	tarend = taridx + len;
	int y = 0;int n_vec_len = 0,n_len =0,n_all_len = 0;
	if(taridx  == n_vec[n_vec_len].begin){//+ n_all_len
		n_len = n_vec[n_vec_len].length;
		
		n_vec_len ++;
		kv_push(match_t,*ml, match_t(0, taridx, n_len));
		tarend = taridx + n_len;
			// n_all_len += n_len;
		y += n_len;
		//continue;
	}else{
		kv_push(match_t,*ml, match_t(refidx-tarlength-1, taridx, len));
		y = 1;
	}
	
	for(y = y; y < tarlength;y ++){//mh-n
		match_t temp = mh->a[y];
		if(temp.taridx  == n_vec[n_vec_len].begin){//+ n_all_len
			n_len = n_vec[n_vec_len].length;
			//n_all_len += n_len;			 
			taridx = temp.taridx ;//+ n_all_len;
			//n_all_len += n_len;
			n_vec_len ++;
			kv_push(match_t,*ml, match_t(0, taridx, n_len));
			tarend = taridx + n_len;
			// n_all_len += n_len;
			y += n_len -1;
			continue;
		}
		//match_t temp = mh->a[y];
		if(tarend >= (temp.taridx + temp.len /*n_all_len*/)){
			continue;
		}else{
			if(temp.len <21){
				continue;
			}else{
				taridx = temp.taridx ;//+ n_all_len;
				refidx = temp.refidx;
				len = temp.len;
				tarend = taridx + len;
				kv_push(match_t,*ml, match_t(refidx-tarlength-1, taridx, len));
				len --;
			}	
			
		}
	}
	sort(ml->a, ml->a + ml->n, comparebylenidx);
	delete [] n_vec;
	int ml_length = ml->n;int z =0;int taridx_n,refidx_n,len_n,ac_z;
	int n_sum = 0;int sum_z = 0;
	////////////////////////
	int mllength =ml_length;
	int tmptaridx,tmprefidx,tmplen;
	int *mld = new int[ml_length];int lcslen = 21;double f = 0.0;
	for(int n = 0;n < ml_length;n ++){
		mld[ n ] = n;
	}
	for( z = 0;z < ml_length;z++){
		match_t cur_temp = ml->a[mld[z]];
		taridx = cur_temp.taridx;
		refidx = cur_temp.refidx;
		tarend = cur_temp.taridx + cur_temp.len ;
		if(cur_temp.len <= lcslen){///////100
			//int mm = 0;
			continue;
		}
		if(sign[taridx] != -1 &&sign[taridx] == sign[tarend - 1] ){
			continue;
		}else if(sign[taridx] == -1 && sign[tarend - 1] == -1){		
			for(int a = taridx ; a < tarend;a ++){
				sign[a] = mld[z];
				//signlen ++;							
			}	
				
			//n_sum ++;	
		}else if(sign[taridx] != -1 && sign[tarend - 1] == -1){		
			int b = sign[taridx];
			match_t pre_temp = ml->a[b];//mld[z]
			int pre_taridx = pre_temp.taridx;
			int pre_len = pre_temp.len;
			int pre_tarend = pre_taridx + pre_len ;
			int rem_len = tarend - pre_tarend;
			if(rem_len <= lcslen){///////100
				continue;
			}
			match_t next_temp = ml->a[mld[z + 1]];
			int next_len = next_temp.len;
			if(rem_len >= next_len){
				ml->a[mld[z]].refidx = refidx + (pre_tarend - taridx);
				for(taridx = pre_tarend; taridx < tarend; taridx++){
					sign[taridx] = mld[z];	
					//signlen ++;				
				}
				
				//n_sum ++;
				ml->a[mld[z]].taridx = pre_tarend;
				ml->a[mld[z]].len = tarend - pre_tarend;
			}else{
				ml->a[mld[z]].taridx = pre_tarend;
				ml->a[mld[z]].refidx = refidx + (pre_tarend - taridx);
				ml->a[mld[z]].len = tarend - pre_tarend;
				tmptaridx = pre_tarend;
				tmprefidx = refidx + (pre_tarend - taridx);
				tmplen = tarend - pre_tarend;;
				sortbymld(z,mllength,tmplen,ml,mld);
				z--;
			}
		}else if(sign[taridx] == -1 && sign[tarend - 1] != -1){
			int b = sign[tarend];
			match_t pos_temp = ml->a[b];
			int pos_taridx = pos_temp.taridx;
			int rem_len = pos_taridx - taridx;
			if(rem_len <= lcslen){/////100
				continue;
			}
			match_t next_temp = ml->a[mld[z + 1]];
			int next_len = next_temp.len;
			if(rem_len >= next_len){
				ml->a[mld[z]].len = pos_taridx - taridx;
				for(taridx  = taridx ; taridx < pos_taridx ; taridx ++){
					sign[taridx] = mld[z];	
					//signlen ++;				
				}
				
				//n_sum ++;
			}else{
				ml->a[mld[z]].taridx = taridx;
				ml->a[mld[z]].refidx = refidx;
				ml->a[mld[z]].len = pos_taridx - taridx;
				sortbymld(z,mllength,ml->a[mld[z]].len,ml,mld);
				z--;
			}
		}else if(sign[taridx] != -1 && sign[tarend - 1] != -1 && sign[taridx] != sign[tarend - 1]){
			
				int a = sign[taridx];int b = sign[tarend];
				match_t next_temp = ml->a[mld[z + 1]];
				match_t pos_temp = ml->a[b];
				match_t pre_temp = ml->a[a];				
				int pre_tarend = pre_temp.taridx + pre_temp.len;
				int pos_taridx = pos_temp.taridx;
				int rem_len = pos_taridx - pre_tarend;
				if(rem_len < lcslen){///////100
					continue;
				}
				if(rem_len > next_temp.len){
					ml->a[mld[z]].refidx = refidx + (pre_tarend - taridx);
					
					for(taridx = pre_tarend; taridx < pos_taridx;taridx++){
						sign[taridx] = mld[z];
						//signlen ++;						
					}
					
					//n_sum ++;
					ml->a[mld[z]].taridx = pre_tarend;
					ml->a[mld[z]].len = pos_taridx - pre_tarend;
				}else{
					ml->a[mld[z]].taridx = pre_tarend;
					ml->a[mld[z]].refidx = refidx + (pre_tarend - taridx);
					ml->a[mld[z]].len = pos_taridx - pre_tarend;
					sortbymld(z,mllength,ml->a[mld[z]].len,ml,mld);
					z--;
				}
		}
	}

	 
	delete [] mld;
	//return ;
}

int main(int argc, char **argv){
	int c;
	int rmq =0;

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
    vector<string> filename = {"chr1.fa", "chr2.fa", "chr3.fa", "chr4.fa", //default chr name list
                    "chr5.fa", "chr6.fa", "chr7.fa", "chr8.fa", "chr9.fa", "chr10.fa", 
                    "chr11.fa", "chr12.fa", "chr13.fa", "chr14.fa", "chr15.fa", "chr16.fa", "chr17.fa", 
                    "chr18.fa", "chr19.fa", "chr20.fa", "chr21.fa", "chr22.fa", "chrX.fa", "chrY.fa"};
	auto startTime = std::chrono::steady_clock::now();
   for(int d = 0; d < 24; d ++){  
		FILE *Rstream;
		FILE *Tstream;
		FILE *Ostream;
		POSITION_RANGE *pos_vec = new POSITION_RANGE[min_size];
		POSITION_RANGE *n_vec = new POSITION_RANGE[min_size];
		char ref[150];char tar[150];char out[150];
		sprintf(ref,"%s%s",argv[argc-3],filename[d].c_str());
		sprintf(tar,"%s%s",argv[argc-2],filename[d].c_str());
		sprintf(out,"%s/%s",argv[argc-1],filename[d].c_str());
		vector<string> taxaref;
		vector<string> taxatar;
		vector<string> Tarsequence;
		vector<string> Refsequence;
		//string seq;
		string upcaseString;
		upcaseString.reserve((1<<16));
		
		//match_v *ml = new *match_v;
		if (argc < 4) {
			fprintf(stderr, "usage: %s file.fasta k\n", argv[0]);
			return EXIT_FAILURE;
		}
		Rstream = fopen(ref, "r");//argv[argc-3]
		Tstream = fopen(tar, "r");//argv[argc-2]
		Ostream = fopen(out, "w");
		if (!Rstream) {
			perror("can't open input Ref file");
			fprintf(stderr, "usage: %s file.fasta k\n", argv[0]);
			return EXIT_FAILURE;
		}

		if (!Tstream) {
			perror("can't open input Tar file");
			fprintf(stderr, "usage: %s file.fasta k\n", argv[0]);
			return EXIT_FAILURE;
		}
		
		//bool flag_n_len = false;
		cout << "starting compressing : " << filename[d]<< endl;
		readRef(Rstream, Refsequence,taxaref);
		//
 		readTar(Tstream, Tarsequence,taxatar,pos_vec,n_vec);	
		//delete [] pos_vec;
		
		int tarlength = Tarsequence.at(0).length();
		match_v *mh = (match_v*)malloc(tarlength *sizeof(match_v));
 		match_v *ml = (match_v*)malloc(tarlength *sizeof(match_v));
		int *sign = new int[tarlength + 100];
		memset(sign, -1 , (tarlength+100)*sizeof(int));
		searchLCS(sign,Tarsequence,Refsequence,mh, ml,pos_vec,n_vec);
		//delete [] mh;
		kv_destroy(mh);
		//free(mh);
		saveDATA(sign,Tstream,Rstream,Ostream,Tarsequence,Refsequence,ml,taxaref,taxatar,upcaseString,out,pos_vec,n_vec);
		//delete [] pos_vec;
		//delete [] n_vec;
		upcaseString.clear();
		//free(ml);
		kv_destroy(mh);
		//int ret = malloc_trim(0);
		Tarsequence.clear();
		Refsequence.clear();
		//free(ml);
	}	 
	  string objfn = argv[argc-1];//"/itmslhppc/itmsl0105/workspace/result/lcs-n.txt";
	  string outfile =argv[argc-1];//"/itmslhppc/itmsl0105/workspace/result";
	  string tarcmd = "tar -cf" + objfn + ".tar -C " + objfn + " . ";
	  system(tarcmd.c_str());
	  string cmd = "rm -rf " + objfn;
	  system(cmd.c_str());
	  
	 // memrgc::bsc::BSC_compress((objfn + ".tar").c_str(), objfn.c_str(), 64);
	  LMSRGC::bsc::BSC_compress((objfn + ".tar").c_str(), objfn.c_str(), 64);
	  auto endTime = std::chrono::steady_clock::now();
	  double compressTime = std::chrono::duration<double>(endTime - startTime).count();
	  cout << "Compress time: " << compressTime << endl;

}
