'''
Logging utilities, used for colored outputs and saving
training/testing data to log files.
===
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
===
'''
# -*- coding: utf-8 -*-
from time import localtime, strftime
import sys, os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class uid:
    moment = strftime("%Y-%b-%d_%H_%M_%S",localtime())

class Log():
    '''
    Class for logging outputs with colors and timestamps
    '''
    def __init__(self, args=None, file_base='output', record=True):
        '''
        Args:
            args (argparse): object with programs arguements
            file_base (string): name of file that logs will be written too
            record (boolean): to record logs to file or not
        '''
        if(not args is None):
            self.log_dir = args.run_dir
            self.log_file = os.path.join(self.log_dir, file_base+uid.moment+'.dat')
            self.record = record
        else:
            self.record = False

    def log(self, str0, rec=True):
        base_str = strftime("[%H:%M:%S]", localtime())
        output_type = "[Output]: "
        print(base_str+output_type+str0)
        if(self.record and rec):
            with open(self.log_file, 'a') as f:
                f.write(base_str+output_type+str0+'\n')

    def info(self, str0):
        base_str = strftime("[%H:%M:%S]", localtime())
        output_type = "[Info]: "
        print(bcolors.OKBLUE+base_str+output_type+str0+bcolors.ENDC)
        if(self.record):
            with open(self.log_file, 'a') as f:
                f.write(base_str+output_type+str0+'\n')

    def success(self, str0):
        base_str = strftime("[%H:%M:%S]", localtime())
        output_type = "[Success]: "
        print(bcolors.OKGREEN+base_str+output_type+str0+bcolors.ENDC)
        if(self.record):
            with open(self.log_file, 'a') as f:
                f.write(base_str+output_type+str0+'\n')

    def warning(self, str0):
        base_str = strftime("[%H:%M:%S]", localtime())
        output_type = "[Warning]: "
        print(bcolors.WARNING+base_str+output_type+str0+bcolors.ENDC)
        if(self.record):
            with open(self.log_file, 'a') as f:
                f.write(base_str+output_type+str0+'\n')

    def error(self, str0):
        base_str = strftime("[%H:%M:%S]", localtime())
        output_type = "[Error]: "
        print(bcolors.FAIL+base_str+output_type+str0+bcolors.ENDC)
        if(self.record):
            with open(self.log_file, 'a') as f:
                f.write(base_str+output_type+str0+'\n')

    # Utils for saving stats to file
    def logTest(self, epoch, testMNLL, testMSE):
        with open('/testErr'+uid.moment+'.dat', 'a') as f:
            f.write('{},\t{}{}\n'.format(epoch, ''.join("{:.6f},\t".format(x) for x in testMNLL), ''.join("{:.6f},\t".format(x) for x in testMSE)))

    def logLoss(self, epoch, loss, trainingMNLL, trainingMSE):
        with open('torchLogs/loss'+uid.moment+'.dat', 'a') as f:
            f.write('{},\t{:.6f},\t{:.6f},\t{:.6f}\n'.format(epoch, loss, trainingMNLL, trainingMSE))

    # Print iterations progress
    # https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
    def print_progress(self, iteration, total, prefix=strftime("[%H:%M:%S]", localtime()), suffix='', decimals=1, bar_length=50):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(bar_length * iteration / float(total)))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()