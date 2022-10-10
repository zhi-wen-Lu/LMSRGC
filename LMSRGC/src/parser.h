#ifndef PARSER_H
#define PARSER_H

#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <ctime>
#include <fstream>
#include <iomanip>

#define FASTA_OK 0
#define FASTA_EOF -1
#define FASTA_ERROR -2

using namespace std;


int parse (FILE *stream, vector<string>& taxa, vector<string>& sequences);
int writeDmat(double** DMat, vector<string>& taxa);

#endif
