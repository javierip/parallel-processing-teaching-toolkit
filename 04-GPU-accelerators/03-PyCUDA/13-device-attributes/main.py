import pycuda.driver as drv

drv.init()
print("%d device(s) found." % drv.Device.count())

for ordinal in range(drv.Device.count()):
    device = drv.Device(ordinal)
    print("Device #%d: %s" % (ordinal, device.name()))
    print("  Compute Capability: %d.%d" % device.compute_capability())
    print("  Total Memory: %s KB" % (device.total_memory() // (1024)))
    collected_device_attributes = [(str(attributes_names), value)
                                   # Python 3.1 and 2.7
                                   for attributes_names, value in device.get_attributes().items()]
    # Python 2.7
    # for att, value in dev.get_attributes().iteritems()]
    collected_device_attributes.sort()

    for attributes_names, value in collected_device_attributes:
        print("  %s: %s" % (attributes_names, value))
