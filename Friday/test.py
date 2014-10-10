def outer():
    x = 0
    for i in xrange(10000000):
        x += i
    y = inner(x)
    return y

def inner(x):
    z = x
    for i in xrange(1000000):
        z += i
    return z

print outer()
