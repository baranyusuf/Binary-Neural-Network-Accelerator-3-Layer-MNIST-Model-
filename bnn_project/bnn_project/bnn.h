#ifndef __BNN_HEADER_
#define __BNN_HEADER_
#include <iostream>

using namespace std;

#include "ap_int.h"
#define DEBUG 1 



//Yoy can change SIZE and function prototype if needed.
typedef  const unsigned int DTYPE; // used as an input type
typedef  int ITYPE;                // used as an output type

const int SIZE = 25;

// ADDED: functions for each layer. 
// This is to allow dataflow pragma later, as well as 
// allow testing of each layer individually
void layer1(const uint32_t input_bits[25], ITYPE out[128]);
void layer2(const ITYPE in[128], ITYPE out[64]);
void layer3(const ITYPE in[64], ITYPE out[10]);

void bnn(const DTYPE IN[SIZE], ITYPE ys[10]);



#endif
