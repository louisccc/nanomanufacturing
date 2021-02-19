import sys
import pyvisa as visa
import libusb1
import usb1
import time

resources = visa.ResourceManager() #Establishes the resource (i.e equipment) manager from PyVISA
funcgen = resources.open_resource('USB0::0xF4EC::0xEE38::SDG2XCAC1L3169::INSTR') #Creates a name for the specific resource and opens the resource. The argument is the resource name specific to the function generator

#The following line can be used to show all the resource names for pieces of equipment connected to the computer
#resources.List_resources()

funcgen.write('C1:BSWV FRQ,1777') #funcgen.write instructs Python to "write" (i.e send) the arguement to the equipment. The argument must be in SCPI
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
