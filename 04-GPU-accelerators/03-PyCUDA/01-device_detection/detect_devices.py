# -*- coding: utf-8 -*-
import pycuda.autoinit
from pycuda import driver


if __name__ == "__main__":

    driver.init()
    devices=driver.Device.count()
    print "N° of Device(s) found" ,devices
    print'-' * 50
    for actual_device in range(driver.Device.count()): 
        dev = driver.Device(actual_device) 
        print "Device #%d: %s" % (actual_device, dev.name()) 
        print " Compute Capability: %d.%d" % dev.compute_capability()     
        print " Total Memory: %s MB" % (dev.total_memory()//(1024)//(1024)) 
        (free,total)=driver.mem_get_info()
        print " Global Memory Occupancy:%f%% Free"%(free*100/total)
        attrs=dev.get_attributes()
        print("\n 	===Attributes for device %d"%devices)
        for (key,value) in attrs.iteritems():
            print("   	 -%s:%s"%(str(key),str(value)))

            print'-' * 50
