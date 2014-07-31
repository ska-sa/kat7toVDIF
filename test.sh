#!/bin/bash
#
# This is an example script
#
#These commands set up the Grid Environment for your job:
#PBS -N PFB_VAILIDATE
#PBS -l nodes=1:ppn=1:series600,walltime=00:59:00
#PBS -q UCTlong
#

# Change to the directory from which the job was submitted.  
cd $PBS_O_WORKDIR

# Print the date and time
echo "Script started at "$(date)

# Host name of the node we are executing on
echo ""
echo "Running on: $(hostname)"
echo "-----------------------------------------------------------------------"

N=1024
P=8
pad=$((N * P * 2))
py_run=/opt/exp_soft/python-2.7.2/bin/python2.7 
test_suite_dir=/home/bhugo/ska-res/ska-ddc/inv_pfb/testSuite
output_directory=/scratch/bhugo/data_out
ipfb_output_directory=/scratch/bhugo

#invoke scripts
$py_run $test_suite_dir/diff_real_inputs.py $output_directory/py_ipfb.dat $ipfb_output_directory/c_ipfb.dat



