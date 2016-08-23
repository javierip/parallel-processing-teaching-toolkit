from multiprocessing import Pool
from time import time
# Import the same addNumbers function we use for the serial example.
from serie import addNumbers


# map requires a function to handle a single argument
def addNumConverter((low, high)):
    return addNumbers(low, high)


def splitRange(low, high):
    # take sub-groups of 1e4 numbers
    STEP = int(1e4)
    r = range(low - 1, high + 1, STEP)
    return zip([x+1 for x in r], r[1:])


if __name__ == '__main__':
    # Use 4 processes.
    pool = Pool(4)

    initialTime = time()

    # We're gonna split the entire range [1, 10^9] into sub groups that can be
    # individually processed.
    subGroups = splitRange(1, 10**9)
    result = sum(pool.map(addNumConverter, subGroups))
    print 'Multiprocessing Example:'
    print 'Result: {}'.format(result)
    print 'Multiprocessing Execution Time: {}s'.format(time() - initialTime)
