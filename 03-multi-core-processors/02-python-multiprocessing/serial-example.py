from time import time


def addNumbers(low, high):
    result = 0
    for number in xrange(low, high+1):
        result += number
    return result

if __name__ == '__main__':
    initialTime = time()
    result = addNumbers(1, 10**9)
    print 'Serial Example:'
    print 'Result: {}'.format(result)
    print 'Serial Execution Time: {}s'.format(time() - initialTime)
