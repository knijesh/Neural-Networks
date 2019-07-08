'''

'__author__' = 'Nijesh'

'''


import csv
import os
import numpy as np
from itertools import izip


class InputFileReader(object):

    """
     Input Reader Class for reading CSV and TSV files. Use the Docstring of get_line()
    """

    def __init__(self, file_path, file_format, num_lines, file_name,skiprows,delimiter):

        """input Instantiator for Reading the files"""

        self.file_path = file_path
        self.file_format = file_format
        self.num_lines = num_lines
        self.file_name = file_name
        self.full_path = os.path.join(self.file_path, self.file_name)
        self.skiprows = skiprows
        self.delimiter = delimiter

    def read_file(self):
        """
        # To get the full list of input file use read_file()
  
        inputfilelist = []
        with open(self.full_path) as infile:
            read = csv.reader(infile, delimiter=self.delimiter)
            for _ in range(self.skiprows):
                next(infile)
            for row in read:
                inputfilelist.append(row)
        return inputfilelist

    def read_line(self):
        """
        Use the Function to get any CSV or TSV file line By line.It emits an iterator and you can use that in a loop
        to get the lines
        :code_usage:
        iters = inputfilereader.read_line() # Class and method Instantiation
        for each in iters:
            Handle each i.e One line at a time
        """
        with open(self.full_path) as csvfile:
            read = csv.reader(csvfile, delimiter=self.delimiter)
            for _ in range(self.skiprows):
                next(csvfile)
            for row in read:
                row = np.array(row)
                yield row

    def _grouped(self,iterable, n):
        """
        Protected Helper Function to get the Pairs or N lines at a time.

        :param iterable: The Sliced List
        :param n: Self.num_lines
        :return: an Iterable
        """
        return izip(*[iter(iterable)] * n)

    def get_lines(self,num_iterations):
        """

        :param num_iterations: The Number of Iterations you would want to fetch N lines at a time
        :return: a generator that can be looped to get N Lines at a time
        :code_usage:
        head = inputfilereader.get_lines(4) # Class and method Instantiation
        for name in head:
            #Handle Name i.e N Lines at a time

        """

        if self.num_lines > 1:
            num_lines = self.num_lines
            output = self.read_file()
            slice_size = num_lines*num_iterations
            output_list = output[:slice_size]
            method = self._grouped(output_list,num_lines)
            return method
        else:
            raise StandardError("For fetching one Value at a time, Please use read_line()")

























