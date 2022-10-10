#include "RMQ_1_n.hpp"

DTidx RMQ_1_n::query(DTidx i, DTidx j) {
	DTidx min = i;
	for (DTidx x = i+1; x <= j; x++)
		if (a[x] < a[min]) min = x;
	return min;
}

/**
 * Standard Constructor. a is the array to be prepared for RMQ.
 * n is the size of the array.
 */
RMQ_1_n::RMQ_1_n(DT* a, DTidx n) {
	this->a = a;
	this->n = n;
}

/**
 * Destructor. Does nothing.
 */
RMQ_1_n::~RMQ_1_n() {
	
}
