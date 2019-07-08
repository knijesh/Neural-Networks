"""

__author__ = "Nijesh"
"""

import argparse
import csv
import os


class CommandLineParser(object):

    def __init__(self, filename ="format_file.txt, filepath ="parser"):
        """
        Args:
            filename:Build ro Score file Autoextracted
            filepath: Filename;Autoextracted

        """
        self.filepath = filepath
        self.filename = filename

    def read_argsfile(self):
        """
       generator to read the files
        """
        full_path = os.path.join(self.filepath, self.filename)
        with open(full_path)as filepointer:
            reader = csv.reader(filepointer, delimiter="\t")
            for read in reader:
                yield read

    def get_attr(self):
        """

        getter function to get the item list
        """
        item_list = []
        args_gen = self.read_argsfile()
        for item in args_gen:
            item_list.append(item)
        header = item_list[0]
        return item_list[1:]

    def set_attr(self):
        """
        generator that yields line which is used by add_argument driver

        """
        getter = self.get_attr()
        for line in getter:
            yield line

    def add_argument(self):
        """

        Main driver method to be invoked for getting the parser args and dictionary in the Neural Network Driver.
        Instantiate the  Class and  use this method
        Usage: parse = CommandLineParser()
                parse.add_argument() #returns a dictionary of arguments with Key as Option name and Value as their
                parsed value.
        """
        global parser, parser
        global requiredNamed, requiredNamed
        parser = argparse.ArgumentParser("\nThe Argument Parser for Artificial Neural Networks\n")
        subparser = parser.add_subparsers(help = "Commands")
        build_parser= subparser.add_parser('build',help="Build Option to Train the Network")
        score_parser = subparser.add_parser('score', help="Score Option to Score the Network")
        requiredNamed = build_parser.add_argument_group("mandatory named arguments")
        scoreNamed = score_parser.add_argument_group("mandatory named arguments")
        dicts = {'String': str, 'Number': float, 'TRUE': True, 'FALSE': False}
        # Subparser for Building Arguments based on first argument {build,score}
        self.filename = self.filename.replace('score', 'build')
        for line in self.set_attr():
            destination = line[1]
            var = destination.replace('-', '_')
            try:
                if not dicts[line[3]]:
                    build_parser.add_argument("-" + line[0], "--" + line[1], action="store",
                                        type=dicts[str(line[-1]).split()[0]], help=line[4], dest=var)
                else:
                    requiredNamed.add_argument("-" + line[0], "--" + line[1], action="store",
                                               type=dicts[str(line[-1]).split()[0]], help=line[4],
                                               required=True, dest=var)
            except IOError, msg:
                parser.error(str(msg))
        # Subparser for Scoring Arguments based on first argument {build,score}
        self.filename = self.filename.replace('build', 'score')
        for line in self.set_attr():
            destination = line[1]
            var = destination.replace('-', '_')
            try:
                if not dicts[line[3]]:
                    score_parser.add_argument("-" + line[0], "--" + line[1], action="store",
                                        type=dicts[str(line[-1]).split()[0]], help=line[4], dest=var)
                else:
                    scoreNamed.add_argument("-" + line[0], "--" + line[1], action="store",
                                               type=dicts[str(line[-1]).split()[0]], help=line[4],
                                               required=True, dest=var)
            except IOError, msg:
                parser.error(str(msg))
        args = parser.parse_args()
        argument = vars(args)
        return argument
