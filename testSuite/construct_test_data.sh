!/bin/bash
#
# This is an example script
#
#These commands set up the Grid Environment for your job:
#PBS -N TEST_DATA_GEN
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
N_longer_filter=$((N*2))
P=8
num_samples_to_use=$((N * P * 5))
output_directory=/home/vusi/ska-ddc-read-only/inv_pfb/testSuite/vusitest
tone_type=noise
py_run=/home/chris/v-env-python-2.7/base/bin/python 
test_suite_dir=ska-ddc-read-only/inv_pfb/testSuite
#don't tweak these formulas unless essential:
#N/2 + 1 non-redundant samples in the fft output of the pfb, by the Hermite-symmetric property of real FFTs, BUT: discard last one due to the SKA infrastructure:
non_redundant_samples=$((N / 2))
non_redundant_pfb_output=$(( (num_samples_to_use / N) * non_redundant_samples ))

if [ -d "$output_directory" ]; then
		rm -r $output_directory		
fi

mkdir $output_directory
stats="Test data characteristics:\n--------------------------------------\nTone type: $tone_type\nN: $N\nP: $P\nNumber of samples in tone: $num_samples_to_use\nNumber of non-redundant samples in pfb output: $non_redundant_pfb_output"
echo -e "--------------------------------------\nDumping test data into: $output_directory\n$stats"

#invoke scripts
$py_run $test_suite_dir/tone_generator.py $output_directory/noise.dat $num_samples_to_use $tone_type $N $P
$py_run $test_suite_dir/filter_generator.py $output_directory/prototype_FIR.dat $N $P
$py_run $test_suite_dir/filter_generator.py $output_directory/long_prototype_FIR.dat $N_longer_filter $P
$py_run $test_suite_dir/pfb_generator.py  $output_directory/noise.dat $output_directory/prototype_FIR.dat $output_directory/pfb.dat $output_directory/unfiltered_ffts.dat $num_samples_to_use $N $P
$py_run $test_suite_dir/ipfb_generator.py $output_directory/pfb.dat $output_directory/prototype_FIR.dat $output_directory/py_ifftedPFB.dat $output_directory/py_ipfb.dat $non_redundant_pfb_output $N $P
echo -e "$stats" > $output_directory/test_data_characteristics.txt

