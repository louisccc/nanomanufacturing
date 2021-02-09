import sys
import pyvisa as visa
import libusb1
import usb1
import time

resources = visa.ResourceManager()
funcgen = resources.open_resource('USB0::0xF4EC::0xEE38::SDG2XCAC1L3169::INSTR')
#print(funcgen.query('*IDN?'))
funcgen.write('C1:BSWV FRQ,1777')
time.sleep(3)
funcgen.write('C1:BSWV FRQ,1995')
time.sleep(3)
funcgen.write('C1:BSWV FRQ,430')
time.sleep(3)
funcgen.write('C1:BSWV FRQ,796')
time.sleep(3)
funcgen.write('C1:BSWV FRQ,23145')
time.sleep(3)
funcgen.write('C1:BSWV FRQ,2021')
