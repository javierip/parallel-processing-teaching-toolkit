# -*- coding: utf-8 -*-

# Parallel Processing Teaching Toolkit
# PyOpenCL - Example 01
# Detect OpenCL Devices
# https://github.com/javierip/parallel-processing-teaching-toolkit

import pyopencl as cl


if __name__ == "__main__":
	print('\n'+'-' *50 + '\nOpenCL Platforms & Devices')
	for platform in cl.get_platforms():
		print('-' * 50)
		print('Platform Name :	' + platform.name)
		print('Platform Vendor :	' + platform.vendor)
		print('Platform Version :	' + platform.version)
		print('Platform Profile :	' + platform.profile)
		print('Platform Extensions :	' + platform.extensions)

		for device in platform.get_devices():
			print('    ' + '-' * 50)
			print('     Device Name:	'  + device.name)
			print('     Device Type:	'  + cl.device_type.to_string(device.type))
			print('     Device Max Clock Speed:	{0} MHz'   .format(device.max_clock_frequency))
			print('     Device Compute Units:	{0}'   .format(device.max_compute_units))
			print('     Device Local Memory:	{0:.0f} KB'   .format(device.local_mem_size/1024.0))
			print('     Device Constant Memory:	{0:.0f} KB'   .format(device.max_constant_buffer_size/1024.0))
			print('     Device Global Memory:	{0:.0f} GB'   .format(device.global_mem_size/1073741824.0))
			print('     Device Max Buffer/Image Size:	{0:.0f} MB'   .format(device.max_mem_alloc_size/1048576.0))
			print('     Device Max Work Group SIze:	{0:.0f}'   .format(device.max_work_group_size))
			print('    ' + '-' * 50)
		print('-' * 50)
		print('\n')



