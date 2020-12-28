import os
import sys
import shutil
import json
import time
import datetime
import tempfile
import warnings
from collections import defaultdict
from typing import Optional
from colorama import Fore
import getpass
import re

from mbpo_pytorch.thirdparty.util import mpi_rank_or_zero

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

LEVEL_INFO = {DEBUG: 'DEBUG', INFO: 'INFO', WARN: 'WARN', ERROR: 'ERROR'}

PROJ_NAME = None
EMAIL_ACCOUNT = 'jiangsy@lamda.nju.edu.cn'
EMAIL_PASSWORD = None
LAST_EMAIL_TIME = None


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'write'), 'Expected file or str, got {}'.format(filename_or_file)
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn('Tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    @classmethod
    def _truncate(cls, string):
        return string[:20] + '...' if len(string) > 23 else string

    def writeseq(self, seq):
        seq = list(seq)
        for elem in seq:
            # do not log color in file
            if self.own_file and is_color(elem):
                continue
            self.file.write(elem)
            if not is_color(elem):
                self.file.write(' ')
        self.file.write('\n')
        self.file.flush()

    def close(self):
        
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        
        self.file = open(filename, 'wt')

    def writekvs(self, kvs):
        for key, value in sorted(kvs.items()):
            if hasattr(value, 'dtype'):
                if value.shape == () or len(value) == 1:
                    # if value is a dimensionless numpy array or of length 1, serialize as a float
                    kvs[key] = float(value)
                else:
                    # otherwise, a value is a numpy array, serialize as a list or nested lists
                    kvs[key] = value.tolist()
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self):
        
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        
        self.file = open(filename, 'w+t')
        self.keys = []
        self.sep = ','

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = kvs.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, key) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(key)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            value = kvs.get(key)
            if value is not None:
                self.file.write(str(value))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        
        self.file.close()


def valid_float_value(value):
    
    try:
        float(value)
        return True
    except TypeError:
        return False


def make_output_format(_format, ev_dir, log_suffix=''):
    
    os.makedirs(ev_dir, exist_ok=True)
    if _format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif _format == 'log':
        return HumanOutputFormat(os.path.join(ev_dir, 'log%s.txt' % log_suffix))
    elif _format == 'json':
        return JSONOutputFormat(os.path.join(ev_dir, 'progress%s.json' % log_suffix))
    elif _format == 'csv':
        return CSVOutputFormat(os.path.join(ev_dir, 'progress%s.csv' % log_suffix))
    else:
        raise ValueError('Unknown format specified: %s' % (_format,))


def logkv(key, val):
    
    Logger.CURRENT.logkv(key, val)


def logkv_mean(key, val):
    
    Logger.CURRENT.logkv_mean(key, val)


def logkvs(key_values):
    
    for key, value in key_values.items():
        logkv(key, value)


def dumpkvs():
    
    Logger.CURRENT.dumpkvs()


def getkvs():
    
    return Logger.CURRENT.name2val


def log(*args, level=INFO, send_email=False):
    if send_email:
        send_mail('\n'.join(map(str, args)), level=level)
    Logger.CURRENT.log(*args, level=level)


def debug(*args, send_email=False):
    log(Fore.WHITE, *args, Fore.WHITE, level=DEBUG, send_email=send_email)


def info(*args, send_email=False):
    log(Fore.GREEN, *args, Fore.RESET, level=INFO, send_email=send_email)


def notice(*args, send_email=False):
    log(Fore.BLUE, *args, Fore.RESET, level=INFO, send_email=send_email)


def warn(*args, send_email=False):
    log(Fore.YELLOW, *args, Fore.RESET, level=WARN, send_email=send_email)


def error(*args, send_email=False):
    log(Fore.RED, *args, Fore.RESET, level=ERROR, send_email=send_email)


def set_level(level):
    Logger.CURRENT.set_level(level)


def get_level():
    return Logger.CURRENT.level


def get_dir():
    return Logger.CURRENT.get_dir()


record_tabular = logkv
dump_tabular = dumpkvs


class ProfileKV:
    def __init__(self, name):
        self.name = "wait_" + name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, _type, value, traceback):
        Logger.CURRENT.name2val[self.name] += time.time() - self.start_time


def profile(name):
    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with ProfileKV(name):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


class Logger(object):
    # A logger with no output files. (See right below class definition)
    #  So that you can still log to the terminal without setting up any output files
    DEFAULT = None  # type: Optional["Logger"]
    # Current logger being used by the free functions above
    CURRENT = None  # type: Optional["Logger"]

    def __init__(self, folder, output_formats):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = folder
        self.output_formats = output_formats

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        if val is None:
            self.name2val[key] = None
            return
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        if self.level == DISABLED:
            return
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


Logger.DEFAULT = Logger.CURRENT = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout)])


def configure(folder=None, format_strs=None, log_email=False, proj_name=None):
    global PROJ_NAME
    PROJ_NAME = proj_name

    if folder is None:
        folder = os.getenv('OPENAI_LOGDIR')
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)
    rank = mpi_rank_or_zero()

    log_suffix = ''
    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv('OPENAI_LOG_FORMAT', 'stdout,log,csv').split(',')
        else:
            log_suffix = "-rank%03i" % rank
            format_strs = os.getenv('OPENAI_LOG_FORMAT_MPI', 'log').split(',')
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(folder=folder, output_formats=output_formats)
    info('Logging to %s' % os.path.realpath(folder))

    if log_email:
        global EMAIL_ACCOUNT
        global EMAIL_PASSWORD

        EMAIL_ACCOUNT = input('Email account:') or EMAIL_ACCOUNT
        EMAIL_PASSWORD = getpass.getpass('Email password:') or EMAIL_PASSWORD
        if not EMAIL_ACCOUNT or not EMAIL_PASSWORD:
            log('Mailing is disabled.')


def reset():
    
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('Reset logger')


class ScopedConfigure(object):
    def __init__(self, folder=None, format_strs=None):
        self.dir = folder
        self.format_strs = format_strs
        self.prevlogger = None

    def __enter__(self):
        self.prevlogger = Logger.CURRENT
        configure(folder=self.dir, format_strs=self.format_strs)

    def __exit__(self, *args):
        Logger.CURRENT.close()
        Logger.CURRENT = self.prevlogger


# ================================================================

def _demo():
    
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    folder = "/tmp/testlogging"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    configure(folder=folder)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see a = 5.5")
    logkv_mean("b", 22.2)
    logkv_mean("b", 44.4)
    logkv("a", 5.5)
    dumpkvs()
    with ScopedConfigure(None, None):
        info("^^^ should see b = 33.3")

    with ScopedConfigure("/tmp/test-logger/", ["json"]):
        logkv("b", -2.5)
        dumpkvs()

    reset()
    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()
    warn("hey")
    error("oh")
    notice('wow')
    logkvs({"test": 1})


def read_json(fname):
    import pandas
    data = []
    with open(fname, 'rt') as file_handler:
        for line in file_handler:
            data.append(json.loads(line))
    return pandas.DataFrame(data)


def read_csv(fname):
    import pandas
    return pandas.read_csv(fname, index_col=None, comment='#')


def is_color(arg):
    if not isinstance(arg, str):
        return False
    return re.match(r'\x1b\[\d+m', arg) is not None


def send_mail(message, level=INFO):
    global LAST_EMAIL_TIME
    cur_time = time.time()
    if LAST_EMAIL_TIME and (cur_time - LAST_EMAIL_TIME < 300):
        return
    if not EMAIL_ACCOUNT or not EMAIL_PASSWORD:
        return

    import smtplib
    from email.mime.text import MIMEText
    from email.header import Header
    mail_user = EMAIL_ACCOUNT
    mail_pass = EMAIL_PASSWORD

    sender = EMAIL_ACCOUNT
    receivers = [EMAIL_ACCOUNT]

    message = MIMEText(message, 'plain', 'utf-8')

    subject = 'Proj:{} {}@{}'.format(PROJ_NAME, LEVEL_INFO[level], time.asctime(time.localtime(time.time())))
    message['Subject'] = Header(subject, 'utf-8')

    try:
        smtpObj = smtplib.SMTP_SSL('smtp.exmail.qq.com:465')
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        LAST_EMAIL_TIME = cur_time
    except smtplib.SMTPException:
        pass


if __name__ == "__main__":
    _demo()
