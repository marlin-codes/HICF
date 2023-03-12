import os
import sys


class Logger(object):
    def __init__(self, logname, now, dataset=None, i=None, analysis='no'):
        if dataset is not None:
            analysis_content = '' if analysis == 'no' else analysis
            print(analysis_content)
            path = os.path.join('log-files/{}/{}'.format(dataset, analysis_content), now.split('_')[0])
            if i is not None:
                path = os.path.join('log-files/{}/{}'.format(dataset, analysis_content), now.split('_')[0], '{}'.format(1000 + i))
        else:
            path = os.path.join('log-files', now.split('_')[0])
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, now.split('_')[1] + '-' + logname + '.txt')
        print('saving log to ', path)

        self.terminal = sys.stdout
        self.file = None

        self.open(path)

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def close(self):
        self.file.close()
