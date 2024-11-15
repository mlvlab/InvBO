import subprocess
import os
import argparse

parser = argparse.ArgumentParser(description="InvBO_Guacamol")
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--task_id', type=str, default='zale')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(args.cuda)

for i in range(10):
    if args.task_id in ['med2', 'zale']:
        alpha = 1000
        delta = 1
    elif args.task_id in ['pdop', 'rano', 'adip', 'valt']:
        alpha = 100
        delta = 0.1
    elif args.task_id in ['osmb']:
        alpha = 100
        delta = 1
    beta = 1
    gamma = 1

    output = subprocess.call(['python3', 'scripts/molecule_optimization.py', 
                                            '--task_id', '{}'.format(args.task_id),
                                            '--track_with_wandb', 'False',
                                            '--wandb_entity', 'ENTITY',
                                            '--alpha', '{}'.format(alpha),
                                            '--beta', '{}'.format(beta),
                                            '--gamma', '{}'.format(gamma),
                                            '--delta', '{}'.format(delta),
                                            'run_invbo',
                                            'done'
                                            ])
