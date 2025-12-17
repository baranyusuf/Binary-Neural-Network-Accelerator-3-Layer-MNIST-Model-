# BNN 

This repo implements a three-layer binar\y neural network optimized for an FPGA.

## Running

Test with `make test`

Generate report with `make report`

Run with `make hls`

## bnn_mnist.py

This generates the header file that our bnn will use to inputs and expected outputs

Run python bnn_mnist.py to generate golden.h
## bnn.cpp

The bnn logic is found in `bnn.cpp` with an entrypoint of `void bnn(const uint32_t input_bits[25], ITYPE ys[10])`.

### Pipeline
```
Input input_bits[] (25 32-bit words, 800 binary inputs with 16 bits of paddint)
|
Layer 1 (128 neurons (25*32b per neuron) -> produces 128 activations
|
Layer 2 (64 neurons (128 inputs; packed as 4*32b) -> produces 64 activations
|
Layer 3 (10 neurons (64 inputs; packed as 2*32b)
|
Output ys[10]
```

## Optimizations

### Overview

- XNOR and popcount instead of MAC operations.
- Split each layer into their own function. This allows us to optimize each layer independantly as well as use the `#pragma HLS DATAFLOW` pragma to pipeline each layer. 
- We optimized for low latency. Thus, our design has a high area footprint and resourse utilization. 


### Layer 1

```
#pragma HLS ARRAY_PARTITION variable=input_bits complete dim=1
#pragma HLS ARRAY_PARTITION variable=results complete
- We array partition our input bits and results to allow for concurrency

#pragma HLS PIPELINE II=1
- Pipelines the neurons so one starts every clock cycle for maximum throughput.

#pragma HLS UNROLL
- Allows us to parallelize the popcount_xnor operations
```

### Layer 2

```
#pragma HLS ARRAY_PARTITION variable=in_packed complete dim=1
- Fully partitions the temporary packed input array so each 32‑bit word can be accessed in parallel during XNOR + popcount operations.

#pragma HLS PIPELINE II=1
- Pipelines the neurons so one starts every clock cycle for maximum throughput.

#pragma HLS UNROLL factor=8
- Unrolls the inner packing loop, enabling multiple bits to be packed simultaneously.

#pragma HLS ARRAY_PARTITION variable=LAYER2 cyclic factor=32 dim=1
- Partitions the weights across 32 ports, allowing multiple neurons to read them concurrently.
```

### Layer 3

```
#pragma HLS ARRAY_PARTITION variable=in_packed complete dim=1
- Partitions in_packed for simultaneous XNOR + popcount operations.

#pragma HLS PIPELINE II=1
- Pipelines the neurons so one starts every clock cycle for maximum throughput.

#pragma HLS UNROLL
- Unrolls the inner packing loop, enabling multiple bits to be packed simultaneously.

#pragma HLS ARRAY_PARTITION variable=LAYER3 complete dim=1
- Partitions LAYER3 so all weights can be accessed in a single cycle.
```

# Testing

Our testbench is found in `bnn_test.cpp`. This file checks the output of each layer against the data in golden.h, ensuring each layer is outputting the correct values.




