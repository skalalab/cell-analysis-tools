.. flim_tools documentation master file, created by
   sphinx-quickstart on Fri Aug 20 13:26:16 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Normalize
=============

Normalizes an array to have values between 0 and 1 by subtrating min value and dividing by new max value.

.. py:function:: normalize(x)

   x: array-like 

   Returns an array normalized to values between 0 and 1

::
   import random

   array = random.random(10,10)
   print(f"")

More Text here 


"""
Normalizes an image to between 0 and 1 by subtrating min value and 
dividing by new max value.

Parameters
----------
im : array-like
    intensity image.

Returns
-------
array-like
    array normalized to 0 and 1.

"""