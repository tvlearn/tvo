from inspect import currentframe, getframeinfo


def print_location():
    frameinfo = getframeinfo(currentframe().f_back)
    print(frameinfo.filename, frameinfo.function, frameinfo.lineno)
