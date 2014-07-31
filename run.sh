#!/bin/sh
#
# This is an example script
#
#These commands set up the Grid Environment for your job:
#PBS -N SKA__IPFB
#PBS -l nodes=srvslsgpu001:ppn=1:seriesGPU,walltime=00:59:00
#PBS -q GPUQ
#

# Change to the directory from which the job was submitted.  
cd $PBS_O_WORKDIR

# Print the date and time
echo "Script started at "$(date)

# Host name of the node we are executing on
echo ""
echo "Running on: $(hostname)"
echo "-----------------------------------------------------------------------"
ipfb_exe=/home/bhugo/ska-res/ska-ddc/inv_pfb/build/inv_pfb
filter_file=/scratch/bhugo/data_out/prototype_FIR.dat
input_file=/scratch/bhugo/data_out/pfb.dat
output_file=/scratch/bhugo/c_ipfb.dat
$ipfb_exe $filter_file $input_file $output_file
