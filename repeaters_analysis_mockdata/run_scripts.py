import sys
import subprocess

nevents = sys.argv[1]

program1 = 'Generate_Repeater_and_UniformBG_Events.py'
program2 = 'Generate_UD_Events.py'
program3 = 'Compute_Tau_Dist.py'

program_list = [program1, program2, program3]
arg_list = [nevents, nevents, '']

for i in range(len(program_list)):
	subprocess.call(['python3',program_list[i],arg_list[i]])
	print('\n***************************\nFinished ' + program_list[i] + '\n***************************\n')
