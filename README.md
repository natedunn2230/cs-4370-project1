# Project 1: Basic Matrix Addition and Multiplication

##### CS-4370-90: Par. Prog. Many GPU's

##### Nathan Dunn

###### Professor Liu

##### 10/04/19

### Compiling and Running

Two source files can be found: 

1. *dunn_project1_add.cu* 
2. *dunn_project1_mult.cu* 

To compile the programs in the Fry environment, run the following commands:

`singularity exec --nv /home/containers/cuda92.sif nvcc dunn_project1_add.cu -o dunn_add`

`singularity exec --nv /home/containers/cuda92.sif nvcc dunn_project1_mult.cu -o dunn_mult`



To run the programs:

`./dunn_add`

`./dunn_mult`

### Editing Matrix and Block Size

At the top of each source file, one can find two define directives: *N* and *BLOCK*. Their default sizes are 8 and 4 respectively. Editing these values and re-compiling the program can be used to ensure correctness of the program.
