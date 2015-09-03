import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy
import random

INPUT_VECTOR_SIZE = 2     # inputs of one neuron
SYNAPSES_VECTOR_SIZE = 2  # synapsesination connections of one neuron
VECTOR_AMOUNT = 4         # number of neurons in a group
MAX_THRESHOLD = 1         # threshold for spiking

def show_configuration():
    print "###################################################"
    print "#  for each neuron:"
    print "#  max number of inputs: %d" % INPUT_VECTOR_SIZE
    print "#  max number of synapses: %d" % SYNAPSES_VECTOR_SIZE
    print "#"
    print "#  total neurons (vectors): %d" % VECTOR_AMOUNT
    print "#  max threshold: %d" % MAX_THRESHOLD
    print "###################################################"

def divide_network_to_groups(d):
    # NOT IMPLEMENTED
    # divide to groups with minimal inter-group connections
    # under maximum group size restriction
    # ref:
    # http://romainbrette.fr/WordPress3/wp-content/uploads/2014/06/BretteGoodman2012.pdf
    #
    groups=[]
    return groups

def debug(title, var):
    print title+':'
    print var
    print "###################################################"

def run():

    show_configuration()

    ################################################################
    #
    #  set weight/input/threshold/destination values for all neurons
    #
    ################################################################

    # weights: between 0.0-1.0 for each of inputs
    # use one vector for weights of a whole neurons group
    w = numpy.random.rand(INPUT_VECTOR_SIZE*VECTOR_AMOUNT)
    w = w.astype(numpy.float32)
    w_gpu = gpuarray.to_gpu(w)

    # inputs: each is 0 or 1
    # use one vector for inputs of a whole neurons group
    #x = numpy.random.randint(2,size=INPUT_VECTOR_SIZE*VECTOR_AMOUNT)
    x = numpy.array([int(1),0,int(1),int(1),0,int(1),0,int(1)])
    #x = numpy.array([0,0,0,0,int(1),int(1),0,0,0,0,0,int(1)])
    #x = numpy.array([int(1),int(1),int(1),int(1),int(1),int(1),int(1),int(1),int(1),int(1),int(1),int(1)])
    # FIXME: optimize to integer/bool?
    x = x.astype(numpy.float32)
    # to gpu
    x_gpu = gpuarray.to_gpu(x)

    # threshold
    t = MAX_THRESHOLD*numpy.random.rand(VECTOR_AMOUNT)
    t = t.astype(numpy.float32)
    threshold_gpu = gpuarray.to_gpu(t)

    # destination (each of the VECTOR_AMOUNT may connect to SYNAPSES_VECTOR_SIZE positions in any x vector)
    # use one vector for destination of a whole neurons group
    d = numpy.empty(VECTOR_AMOUNT*SYNAPSES_VECTOR_SIZE)
    d.fill(-1)
    for i in range(VECTOR_AMOUNT): # run over all neurons
        connections_num = random.randint(0,SYNAPSES_VECTOR_SIZE) # choose how many connections for each
        j=0
        while j < connections_num: # choose randomly connections_num connections
            d[i*SYNAPSES_VECTOR_SIZE+j] = random.randint(0,INPUT_VECTOR_SIZE*VECTOR_AMOUNT-1) # optional connection
            was_chosen_already=False
            for k in range(j): # verify that this connection was not chosen yet
                if d[i*SYNAPSES_VECTOR_SIZE+k] == d[i*SYNAPSES_VECTOR_SIZE+j]:
                    was_chosen_already=True
            if was_chosen_already == False:
                j+=1
    # FIXME: set some destination to different groups/blocks
    #        by adding MAX_GROUP_SIZE*block_number to the d_value. Then d_value % MAX_GROUP_SIZE gives the location
    #        in the block and d/MAX_GROUP_SIZE gives the block number.

    # FIXME: optimize to integer/bool?
    d = d.astype(numpy.float32)
    synapses_gpu = gpuarray.to_gpu(d)

    # prepare vectors for results:

    # weighted sum
    weighted_sum_gpu = gpuarray.zeros(VECTOR_AMOUNT, dtype=numpy.float32)

    # fired
    fired_gpu = gpuarray.zeros(VECTOR_AMOUNT, dtype=numpy.float32)


    ################################################################
    #
    #  declare kernel
    #
    ################################################################

    mod = SourceModule("""
      #include <stdio.h>
      #define INPUT_VECTOR_SIZE 2
      #define SYNAPSES_VECTOR_SIZE 2
      #define RUN_CYCLES 4

      __device__ void sigma(float *x, float *w, float *weighted_sum, float *threshold, float *fired)
      {
        int idx = blockIdx.x *blockDim.x + threadIdx.x;
        int vec_num = (idx - idx % INPUT_VECTOR_SIZE)/INPUT_VECTOR_SIZE;

        //printf("input x in place %d is %f\\n", idx,  x[idx]);
        //printf("weight in place %d is %f\\n", idx,  w[idx]);
        atomicAdd(&weighted_sum[vec_num], x[idx]*w[idx]);
        if(weighted_sum[vec_num]>=threshold[vec_num]) {
          printf("weighted sum of vector %d: %f\\n", vec_num, weighted_sum[vec_num]);
          printf("threshold of vector %d: %f\\n", vec_num, threshold[vec_num]);
          fired[vec_num]=1.0;
          weighted_sum[vec_num]=0;
        }
      }

      __device__ void zero(float *x)
      {
        int idx = blockIdx.x *blockDim.x + threadIdx.x;
        x[idx]=0;
        //printf("zero %d\\n", idx);
      }

      __device__ void update_inputs(float *fired, float *synapses, float *x)
      {
        int idx = blockIdx.x *blockDim.x + threadIdx.x;
        int i;
        int j;

        if (fired[idx] == 1) { // update fired only
          i = 0;
          while ((synapses[idx*SYNAPSES_VECTOR_SIZE + i] > -1) && (i<SYNAPSES_VECTOR_SIZE)) {
            // run on all outputs from fired neuron
            j = synapses[idx*SYNAPSES_VECTOR_SIZE + i];
            printf("update input number %d on step %d within thread %d\\n", j, i, idx);
            x[j]=1; // update the corresponding input in destination neuron
            i+=1;
          }
        }
      }

      __global__ void cycle(float *x, float *w, float *weighted_sum, float *threshold, float *fired, float *synapses)
      {
        int i;

        for(i=0;i<RUN_CYCLES;i++) {
          //printf("zero %d\\n", i);
          zero(fired);
          __syncthreads();
          //printf("sigma %d\\n", i);
          sigma(x, w, weighted_sum, threshold, fired);
          __syncthreads();
          //printf("zero %d\\n", i);
          zero(x);
          __syncthreads();
          //printf("update %d\\n", i);
          update_inputs(fired, synapses, x);
          __syncthreads();
        }
      }

      """)


    ################################################################
    #
    #  debug before running kernel
    #
    ################################################################

    debug("inputs",x)
    debug("weights",w)
    debug("inputs before running network",x_gpu.get())
    debug("thresholds", threshold_gpu.get())
    debug("synapses of all %d neurons/vectors" % VECTOR_AMOUNT, synapses_gpu.get())

    ################################################################
    #
    #  running kernel
    #
    ################################################################

    cycle = mod.get_function("cycle")
    cycle(x_gpu, w_gpu, weighted_sum_gpu, threshold_gpu, fired_gpu, synapses_gpu, block=(INPUT_VECTOR_SIZE*VECTOR_AMOUNT,1,1))

    ################################################################
    #
    #  debug after running kernel
    #
    ################################################################

    debug("last weighted sum results", weighted_sum_gpu.get())
    debug("last fired neurons", fired_gpu.get())
    debug("inputs after running network", x_gpu.get())


if __name__ == "__main__":
    run()
