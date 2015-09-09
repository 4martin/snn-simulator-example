#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from scipy.sparse import *
import numpy
import random
import sys

INPUT_VECTOR_SIZE = 2     # inputs of one neuron
SYNAPSES_VECTOR_SIZE = 2  # synapsesination connections of one neuron
NEURONS_IN_GROUP = 4      # number of neurons in a group
MAX_THRESHOLD = 1         # threshold for spiking
GROUPS_AMOUNT = 2         # number of neurons groups (correspond to blocks on the GPU)

def show_configuration():
    print "###################################################"
    print "#  for each neuron:"
    print "#  max number of inputs: %d" % INPUT_VECTOR_SIZE
    print "#  max number of synapses: %d" % SYNAPSES_VECTOR_SIZE
    print "#"
    print "#  neurons in a group: %d" % NEURONS_IN_GROUP
    print "#  number of groups: %d" % (GROUPS_AMOUNT)
    print "#  total neurons: %d" % (NEURONS_IN_GROUP*GROUPS_AMOUNT)
    print "#  max threshold: %d" % MAX_THRESHOLD
    print "###################################################"

def debug(title, var):
    print title+':'
    print var
    print "###################################################"

def divide_network_to_groups():
    # NOT IMPLEMENTED
    # divide to groups with minimal inter-group connections
    # under maximum group size restriction (block size in the GPU)
    # It is a graph-cut problem - graph partitioning optimizing edges cut to minimum
    # while satisfying additional conditions.
    #
    # ref:
    # http://romainbrette.fr/WordPress3/wp-content/uploads/2014/06/BretteGoodman2012.pdf
    #


    # instead - an example network with GROUPS_AMOUNT dense groups and minor inter-group connection is built:

    # create all groups
    GI=NEURONS_IN_GROUP*INPUT_VECTOR_SIZE # group inputs
    GS=NEURONS_IN_GROUP*SYNAPSES_VECTOR_SIZE # group synapses
    g = numpy.zeros((GI*GROUPS_AMOUNT,GS*GROUPS_AMOUNT)) # large (sparse) matrixi
    g = g.astype(numpy.float32)

    # weights: between 0.0-1.0 for each of inputs
    # indices:
    # (rows) input#, (columns) synapse#

    # inside group connections:
    for i in range(GROUPS_AMOUNT):
      g[0+i*GI,6+i*GS]=0.2 # on group #0, #0 synapse of neuron #3 connects to #0 input of neuron #0 with weight 0.2
      g[1+i*GI,2+i*GS]=0.6
      g[2+i*GI,7+i*GS]=0.5
      g[3+i*GI,4+i*GS]=0.7
      g[4+i*GI,0+i*GS]=0.4
      g[6+i*GI,5+i*GS]=0.8

      #g[5+i*GI,1+i*GS]=0.7123

    # inter-group connections
    # group 1 depends on group 0
    g[7+1*GI,3+0*GS]=0.9  # #1 synapse of neuron #2 in group #0 connects to #1 input of neuron #3 in group #1

    numpy.set_printoptions(linewidth=10000)
    print g
    return g

def get_weights_graph():

    # Assuming that the connection matrix is sparse, the data 
    # structure used is compressed Sparse Row/Column matrix.
    # The CSR high efficiency of rows are used for the weights to target neurons,
    # to achieve coalesced memory access during spike distribution.
    # http://homepages.cwi.nl/~sbohte/publication/slazynski2012network.pdf
    #

    # A dense representation haa NEURONS_IN_GROUP*SYNAPSES_VECTOR_SIZE
    # columns and NEURONS_IN_GROUP*INPUT_VECTOR_SIZE rows, each stating the
    # the corresponding wight or a zero for no connection. Each neuron spans over
    # SYNAPSES_VECTOR_SIZE columns and INPUT_VECTOR_SIZE rows.
    # Groups of neurons (more dense connections) are located in neighbour indices, so
    # they land in the same block letting them run for longer periods while using
    # shared memory, until they need to connect to another group which runs on
    # a different block.
    #
    #                            neuron synapses  X
    #   ----------------------------------------- >
    #   |██████| |        |        |        |   
    #   |██████| |        |        |        |   
    #   |██████| |        |        |        |   
    #   |██████| |        |        |        |   
    #   |------- |        |        |        |   
    #   |--------|--------|--------|--------|----
    #   |     .  |███|    |        |  .     |   
    # n |        |----    |        |        |   
    # e |        |        |        |        |   
    # u |        |        |        |        |   
    # r |--------|--------|--------|--------|----
    # o |        |  .     |██████| |        |   
    # n |        |        |██████| |  .     |   
    #   |        |        |██████| |        |   
    # i |        |        |------- |        |   
    # n |--------|--------|--------|--------|----
    # p |    .   |        |        |█████|  |   
    # u |        |        |        |█████|  |   
    # t |        |        |        |------  |   
    # s |        |        |        |        |   
    #   |--------|--------|--------|--------|----
    # Y v
    #
    # This is a Weights matrix (W):
    # =============================
    # Each of the large squares (16) represents synapses of neurons group (on axis X) connecting
    # to inputs of neurons group (on axis Y). 
    # On the diagonal there are (smaller) squares representing (dense) connections inside
    # a group. The dots on other squares represent inter-group connections.
    # The matrix is splitted to vertical slices, each containing neurons with synapses from one group.
    # Each group runs later on a separate GPU block.
    # When a spike goes to a neuron in another block there is a mechanism that updates the required block.
    #
    # The CSR representation of the above matrix is:
    # A - an array of all non-zero weights (right to left, top down)
    # B - an array where value in place i is the A-index of the first non-zero number on row i of W.
    #     The size |A| is added to B.
    # C - an array of the column indices in W of each of A items.
    #
    # A block that has dependency needs to get periodic approvals until which clock step it
    # may run. A bidirectional dependency between blocks can be solved by running each time
    # during some fixed clock slices (e.g. 1000 clocks). If no spikes were done, just continue
    # with the next slice. If a spike was emitted, cut the slice to 1/2 and repeat calculation
    # on both blocks. Update the corresponding spike as needed.
    #

    groups=divide_network_to_groups()
    CSC_groups=[]
    CSC_vectors_lengths=numpy.zeros(3*GROUPS_AMOUNT, dtype=numpy.float32)
    CSC_vectors_start_index=numpy.zeros(3*GROUPS_AMOUNT, dtype=numpy.float32)

    # split large matrix to GROUPS_AMOUNT group slices
    for i in range(GROUPS_AMOUNT):
        g_slice=groups[:,i*SYNAPSES_VECTOR_SIZE*NEURONS_IN_GROUP:(i+1)*SYNAPSES_VECTOR_SIZE*NEURONS_IN_GROUP]
        #print "slice ...."
        #print g_slice

        m=csc_matrix(g_slice)
        A=m.data
        B=m.indptr
        C=m.indices
        #print A,B,C
        # keep vector (of CSC representation for each group) lengths
        CSC_vectors_lengths[0+i*3]=len(A)
        CSC_vectors_lengths[1+i*3]=len(B)
        CSC_vectors_lengths[2+i*3]=len(C)
        #print "CSC_vectors_lengths ", CSC_vectors_lengths
        if i<(GROUPS_AMOUNT-1):
            # check on which location each vector begins
            # next vector begins at the previous location + its vector length
            # this is needed for in-kernel vectors usage optimization
            CSC_vectors_start_index[0+(i+1)*3]=CSC_vectors_start_index[0+i*3]+len(A)
            CSC_vectors_start_index[1+(i+1)*3]=CSC_vectors_start_index[1+i*3]+len(B)
            CSC_vectors_start_index[2+(i+1)*3]=CSC_vectors_start_index[2+i*3]+len(C)
            #print "CSC_vectors_start_index ", CSC_vectors_start_index

        CSC_groups.append([A,B,C])

    return CSC_groups,CSC_vectors_start_index,CSC_vectors_lengths

def run():

    show_configuration()

    # get network
    CSC_groups,CSC_vectors_start_index,CSC_vectors_lengths=get_weights_graph()

    # concat all CSC vectors to simplify load to GPU
    # calculate total lengths
    concat_vectors_lengths=numpy.zeros(3, dtype=numpy.float32)
    j=0
    for i in CSC_vectors_lengths:
         concat_vectors_lengths[j%3]+=CSC_vectors_lengths[j] # calculate total lengths for all A,B,C
         j+=1

    # allocate concatenated vectors
    ccA=numpy.zeros(concat_vectors_lengths[0], dtype=numpy.float32)
    ccB=numpy.zeros(concat_vectors_lengths[1], dtype=numpy.float32)
    ccC=numpy.zeros(concat_vectors_lengths[2], dtype=numpy.float32)

    # concating all A in to ccA, B to ccB and C to ccC
    ccA_counter=0
    ccB_counter=0
    ccC_counter=0
    for i in range(GROUPS_AMOUNT):
        A,B,C = CSC_groups[i]
        for j in range(CSC_vectors_lengths[0+i*3]): # run over each A length
            ccA[j+ccA_counter]=A[j]
        ccA_counter=j+1
        for j in range(CSC_vectors_lengths[1+i*3]): # run over each B length
            ccB[j+ccB_counter]=B[j]
        ccB_counter=j+1
        #print "range: ",CSC_vectors_lengths[2+i*3]
        for j in range(CSC_vectors_lengths[2+i*3]): # run over each C length
            #print "ccC index is ", j, " writing ",C[j]
            ccC[j+ccC_counter]=C[j]
        ccC_counter=j+1

    #print "==============> ",concat_vectors_lengths
    #print "==============> ",ccA
    #print "==============> ",ccC

    # more data structures:
    # =====================
    # Inputs - array. Size according to block size limit from weight matrix.
    # Threshold - array per neuron (small).
    # Action Potential (AC) - array. Size according to block size limit from weight matrix.
    # Fired - array per neuron (small). 
    # Cross block dependency - matrix per block (small).

    # inputs: each is 0 or the corresponding weight
    # use one vector for inputs of a whole neurons group

    X = numpy.array([0.2,0,0.5,0.7,0.4,0,0,0.9,0.2,0,0.5,0.7,0.4,0,0,0.9])
    #X = numpy.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    X = X.astype(numpy.float32)

    # threshold
    TH = MAX_THRESHOLD*numpy.random.rand(NEURONS_IN_GROUP*GROUPS_AMOUNT)
    #TH = MAX_THRESHOLD*numpy.zeros(NEURONS_IN_GROUP*GROUPS_AMOUNT)
    TH = TH.astype(numpy.float32)

    # to gpu
    # currently one one group is loaded
    # to load the full ndarray, the following can be used:
    # http://documen.tician.de/pycuda/array.html#pycuda.gpuarray.GPUArray.set
    ccA_gpu = gpuarray.to_gpu(ccA)
    ccB_gpu = gpuarray.to_gpu(ccB)
    ccC_gpu = gpuarray.to_gpu(ccC)
    X_gpu = gpuarray.to_gpu(X)
    TH_gpu = gpuarray.to_gpu(TH)

    # CSC_vectors_start_index and CSC_vectors_lengths of CSC vectors
    CSC_vectors_lengths_gpu = gpuarray.to_gpu(CSC_vectors_lengths)
    CSC_vectors_start_index_gpu = gpuarray.to_gpu(CSC_vectors_start_index)

    # prepare vectors for results:
    # weighted sum
    AC_gpu = gpuarray.zeros(NEURONS_IN_GROUP*GROUPS_AMOUNT, dtype=numpy.float32)

    # fired
    fired_gpu = gpuarray.zeros(NEURONS_IN_GROUP*GROUPS_AMOUNT, dtype=numpy.float32)

    ################################################################
    #
    #  declare kernel
    #
    ################################################################

    kernel_code_template = """
      #include <stdio.h>
      #define INPUT_VECTOR_SIZE 2
      #define SYNAPSES_VECTOR_SIZE 2
      #define NEURONS_IN_GROUP 4
      #define GROUPS_AMOUNT 2
      #define INPUTS_PER_GROUP (INPUT_VECTOR_SIZE*NEURONS_IN_GROUP)
      #define GROUP_NUMBER_MASK (INPUTS_PER_GROUP*(GROUPS_AMOUNT-1))
      #define MAX_GROUP_UPDATE_QUEUE_LEN 8 // must be 2^n to work with modulo optimization (see atomicAnd below)
      #define PERIODIC_UPDATE_CYCLES 4
      #define UPDATE_PERIODS 1

      // management of inter-group updates on shared memory
      __device__ struct update_group_entry {
        int clock; // Note: add __padding for alighnment if using 64 bit float
        int input;
        float weight;
      } group_updates_queue[GROUPS_AMOUNT][MAX_GROUP_UPDATE_QUEUE_LEN];
      __device__ int first_on_queue[GROUPS_AMOUNT]; // mod MAX_GROUP_UPDATE_QUEUE_LEN
      __device__ int already_on_queue[GROUPS_AMOUNT];
      volatile __device__ int safe_clock[GROUPS_AMOUNT];

/*
 *   # neural state update + spike generation:
 *   # =======================================
 *   # each input has one of 2 values - 0 or the corresponding weight.
 *   # each group/block verifies that it is safe to run for the current clock.
 *   # safe means that if there is dependency on another group - the other block signals updates for inputs
 *   # on current block at certain clocks, or alternatively no updates until some recent clock.
 *   # block run on all these inputs of neurons in current block, compare to threshold, and update fired
 *   # array. When done, zero all inputs (assumption of 1 clock decay of the spike).
 */

      __device__ void sigma(float *X, float *AC, float *TH, float *fired, uint clock)
      {
        const uint tx = threadIdx.x;
        const uint bx = blockIdx.x;
        const uint vec_num = tx/INPUT_VECTOR_SIZE+bx*NEURONS_IN_GROUP;
        int first_index;

        // busy loop if no "safe" clock in the future
        if(bx==1){ // FIXME: condition should be "is dependent group?"
          if (clock>safe_clock[bx]) {
            printf("busy loop on block %d clock %d before safe %d\\n", bx, clock, safe_clock[bx]);
          } else {
            printf("skip busy as clock %d before safe %d\\n", clock, safe_clock[bx]);
          }
          while(clock>safe_clock[bx]) {
            // busy wait
            // maybe some variation on _gpu_sync() could be used here.
            // http://fulmanski.pl/zajecia/cuda/zajecia_20122013/materialy/TR_GPU_synchronization.pdf
            printf("%d, ",clock);
          }
        }

        if (already_on_queue[bx] > 0) { // must update inputs due to spikes from other groups
          printf("handling queue for group %d length of %d at clock %d\\n", bx, already_on_queue[bx], clock);
          first_index=first_on_queue[bx];
          printf("on queue index %d, clock %d, input %d, weight %f\\n", first_index, group_updates_queue[bx][first_index].clock, group_updates_queue[bx][first_index].input, group_updates_queue[bx][first_index].weight);
          if(clock==group_updates_queue[bx][first_index].clock) {
            // update the input using the values from the queue
            X[group_updates_queue[bx][first_index].input]=group_updates_queue[bx][first_index].weight;
          }
          atomicAdd(&already_on_queue[bx],-1); // FIXME: take care with parallel changes (consider A Parallel Counter Class - http://www.drdobbs.com/parallel/atomic-operations-and-low-wait-algorithm/240160177)
          atomicAdd(&first_on_queue[bx],1);
          atomicAnd(&first_on_queue[bx],MAX_GROUP_UPDATE_QUEUE_LEN-1); // next on cyclic buffer - optimization of modulo (no problem after previous atomic add, since during the transition
                                                                       // between MAX_GROUP_UPDATE_QUEUE_LEN-1 to MAX_GROUP_UPDATE_QUEUE_LEN, these are orthogonal bits)
        }

        if (tx<INPUT_VECTOR_SIZE*NEURONS_IN_GROUP) {

          atomicAdd(&AC[vec_num], X[tx+bx*INPUT_VECTOR_SIZE]);

          if(AC[vec_num]>=TH[vec_num]) {
            fired[vec_num]=1.0; // it is written over INPUT_VECTOR_SIZE times
            printf("fired[%d]=%f on clock %d\\n", vec_num, fired[vec_num], clock);
          } else {
            //printf("under TH of fired[%d]=%f\\n", vec_num, fired[vec_num]);
          }
        }
      }

      __device__ void zero(float *x)
      {
        const uint tx = blockIdx.x *blockDim.x + threadIdx.x;
        if (tx<INPUT_VECTOR_SIZE*NEURONS_IN_GROUP) {
            x[tx]=0;
        }
      }

/*
 *   # spike distribution:
 *   # ===================
 *   # inside a block, run on the weights with a coalesced memory access, multiply by corresponding 
 *   # fired array (the indices derived from C by [floor of] division to INPUT_VECTOR_SIZE). Update the
 *   # corresponding input (the indices are in C). When done, zero all fired array (assumption of
 *   # 1 clock decay of the spike).
 *   # Note: An attempt to update another group (block) is done using group_updates_queue mechanism.
 */


      __device__ void update_inputs(float *ccA, float *ccC, float *fired, float *X, float *CSC_vectors_start_index, float *CSC_vectors_lengths, uint clock)
      {
        const uint tx = threadIdx.x;
        const uint bx = blockIdx.x;

        int a_len_index=0+bx*3;
        int c_len_index=2+bx*3;
        int a_index=tx+CSC_vectors_start_index[a_len_index];
        int c_index=tx+CSC_vectors_start_index[c_len_index];
        int input_index = ccC[c_index];
        int fired_index = input_index/SYNAPSES_VECTOR_SIZE; // neuron number
        int input_group=(input_index&GROUP_NUMBER_MASK)/INPUTS_PER_GROUP; // to which block goes the index

        //printf("BLOCK %d\\n", bx);

        if(tx<CSC_vectors_lengths[a_len_index]) { // running over (the relevat subarray of) A
          //printf("block %d, input_index %d, MASK %x, GROUP NUM %d\\n", bx, input_index, GROUP_NUMBER_MASK, (input_index&GROUP_NUMBER_MASK)/INPUTS_PER_GROUP);
          if(input_group==bx) { // updating current group
            X[input_index] = ccA[a_index]*fired[fired_index];
            printf("normal update in block %d for %d with %f\\n",bx, input_index, ccA[a_index]*fired[fired_index]);
          } else { // must update a different group
            if(fired[fired_index]>0.0) { // ignore on non fired neuron
              printf("external update in block %d at clock %d for input %d with fired_index %d fire %f tell block %d\\n",bx, clock, input_index, fired_index, fired[fired_index], input_group);
              if(already_on_queue[input_group]<MAX_GROUP_UPDATE_QUEUE_LEN) {
                group_updates_queue[input_group][first_on_queue[input_group]].clock=clock;
                group_updates_queue[input_group][first_on_queue[input_group]].input=input_index;
                group_updates_queue[input_group][first_on_queue[input_group]].weight=ccA[a_index]*fired[fired_index];
                already_on_queue[input_group]+=1;
              } else {
                printf("QUEUE TOO LONG on group %d! Spike will be ignored!!!\\n", input_group);
              }
            }
          }
          printf("tx %d fired[%d] %f ccA %f X %f\\n", tx, fired_index, fired[fired_index], ccA[a_index], X[input_index]);
        }
      }

      __global__ void cycle(float *X, float *ccA, float *ccB, float * ccC, float *AC, float *TH, float *fired, float *CSC_vectors_start_index, float *CSC_vectors_lengths)
      {
        uint clock;
        uint periods;

        //if(blockIdx.x==0) {
        //    return;
        //}


        for(periods=0;periods<UPDATE_PERIODS;periods++) {
          for(clock=0+periods*PERIODIC_UPDATE_CYCLES;clock<PERIODIC_UPDATE_CYCLES*(periods+1);clock++) {
            zero(fired);
            zero(AC);
            __syncthreads();
            sigma(X, AC, TH, fired, clock);
            __syncthreads();
            zero(X);
            __syncthreads();
            update_inputs(ccA, ccC, fired, X, CSC_vectors_start_index, CSC_vectors_lengths, clock);
            __syncthreads();
          }
          //printf("PERIOD %d\\n", periods);
          if(blockIdx.x==0){ // FIXME: condition should be "is non-dependant group?"
            if (already_on_queue[1] == 0) {
              safe_clock[1]=clock; // FIXME: atomic? clock-1?
              printf("update clean SAFE to clock %d\\n", safe_clock[1]);
            } else {
              safe_clock[1]=group_updates_queue[1][first_on_queue[1]].clock; // FIXME: atomic? clock-1?
              printf("update dirty SAFE to clock %d\\n", safe_clock[1]);
            }
          }
        }
      }
      """



    kernel_code = kernel_code_template

    mod = SourceModule(kernel_code)

    ################################################################
    #
    #  debug before running kernel
    #
    ################################################################

    debug("inputs",X)
    debug("thresholds", TH_gpu.get())

    ################################################################
    #
    #  running kernel
    #
    ################################################################

    cycle = mod.get_function("cycle")
    cycle(X_gpu, ccA_gpu, ccB_gpu, ccC_gpu, AC_gpu, TH_gpu, fired_gpu, CSC_vectors_start_index_gpu, CSC_vectors_lengths_gpu, block=(SYNAPSES_VECTOR_SIZE*NEURONS_IN_GROUP,1,1), grid=(GROUPS_AMOUNT,1))

    ################################################################
    #
    #  debug after running kernel
    #
    ################################################################

    debug("last fired neurons", fired_gpu.get())
    debug("inputs after running network", X_gpu.get())


if __name__ == "__main__":
    run()



    #
    # improvement options to examine:
    # ===============================
    # parallel sum during AC calculation (complexity drop fron o(n) to o(log n), but maybe for
    # such small input amounts per neuron it doesn't make sense.
    #
    # loop unrolling. 
