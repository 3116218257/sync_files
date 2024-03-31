import datetime
import argparse

def create_exp_name(comments=''):
    x = datetime.datetime.now()
    name = x.strftime("%y%m%d") + '-' + x.strftime("%X")
    if len(comments) > 0:
        name = comments + '_' + name
    return name
