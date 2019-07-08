"""

__author__ ='Nijesh'

"""

import os
import pandas as pd
import numpy as np
import json
import re
from itertools import izip_longest


class ValidationError(Exception):
    def __init__(self,message):
        super(ValidationError, self).__init__(message)

    def __repr__(self):
        return self.message


class WeightValidate(object):
    """
    The Class to Validate Weight File
    """

    def __init__(self, path, filename, filetype='csv'):
        """

        :param path: The File Path
        :param filename: The FIle Name of Weight File
        :param filetype: Type of Weight file;Defaulted to csv
        """
        self.path = path
        self.filename = filename
        self.full_path = os.path.join(self.path, self.filename)
        self.filetype = filetype

    def weightfile_nan(self):
        """
        NaN Validation Method
        :return: False if NAN Validation is  Successful
        """
        df = pd.read_csv(self.full_path)
        weights_col = df['Weight']
        result = weights_col.isnull().values.any()
        index = df['Weight'].index[df['Weight'].apply(np.isnan)]
        df_index = df.index.values.tolist()
        nan_rows = [df_index.index(i) for i in index]
        if result:
            raise ValidationError("NaN Validation Failed. Weight File has Empty/NaN/Invalid entries "
                                  "at Indices {0}".format(nan_rows))

        else:
            return True

    @staticmethod
    def get_network_struct():
        """
        Static Method for getting network structure
        :return: shape of network from networks.txt file
        """
        networkpath = os.path.abspath('../data/DNN.txt')
        with open(networkpath, 'r') as jsonfile:
            content = json.load(jsonfile)
            list_layers = content['layers']
            input_layer = list_layers[0]
            output_layer = list_layers[-1]
            num_input_nodes = input_layer['Input']['nodes']
            num_output_layer_nodes = output_layer['Output']['nodes']
            list_layer = list_layers[1:-1]
            hlayers = []
            hlayernodes = []
            dicts ={'false':False ,'true':True}
            for each in list_layer:
                for key in each.keys():
                    match = re.match("^HL[0-9]$", key)
                    if match.group(0):
                        hlayers.append(str(match.group(0)))
            hiddenbias = []
            for i, layer in enumerate(hlayers):
                temp = list_layer[i]
                num_nodes = temp[layer]['nodes']
                bias = temp[layer]['addBias'].lower()
                bias = dicts[bias]
                hiddenbias.append(bias)
                if not bias:
                    hlayernodes.append(int(num_nodes))
                else:
                    hlayernodes.append(int(num_nodes)+int(1))
            input_bias = input_layer['Input']['addBias'].lower()
            input_bias = dicts[input_bias]
            output_bias = output_layer['Output']['addBias'].lower()
            output_bias = dicts[output_bias]
            if input_bias:
                num_input_nodes = int(num_input_nodes)+int(1)
            if output_bias:
                num_output_layer_nodes = int(num_output_layer_nodes) + int(1)

            shape = (int(num_input_nodes), hlayernodes, int(num_output_layer_nodes))
            biases = (input_bias,hiddenbias,output_bias)
            return shape,biases

    def weight_count_validate(self):
        """
        WeightCount Validation for Weight MAtrices at each intermediate layer
        :return: True if its Successful
        """
        # Node Count from Network Structure File
        global hidden_diff, hidden_diff
        shape,biases = WeightValidate.get_network_struct()
        input_node = shape[0]
        output_node = shape[-1]
        hidden_nodes = shape[1:-1][0]
        df = pd.read_csv(self.full_path)
        source_layer, source_node, = df['Source Layer'], df['Source Node'],
        destination_layer, destination_node, weights = df['Destination Layer'], df['Destination Node'], df['Weight']
        hiddenbias = biases[1]
        if hiddenbias[0]:
            input_matrix_num_weights = (input_node) * (hidden_nodes[0]-1)
        else:
            input_matrix_num_weights = (input_node) * (hidden_nodes[0])
        output_matrix_num_weights = (hidden_nodes[-1]) * output_node
        hidden_matrix_num_weights = []
        for i, num in enumerate(hidden_nodes):
            try:
                bias_for_index_next = hiddenbias[i+1]
                if bias_for_index_next:
                    hidden_weights = num * (hidden_nodes[i + 1]-1)
                else:
                    hidden_weights  = num * (hidden_nodes[i + 1])
                hidden_matrix_num_weights.append(hidden_weights)

            except IndexError:
                pass

        network_file_output = (input_matrix_num_weights, hidden_matrix_num_weights, output_matrix_num_weights)

        # Input Node Count from Weights File
        input_df_len = len(df[source_layer == 0]['Weight'])
        dest = destination_layer.max()
        output_df_len = len(df[destination_layer == dest]['Weight'])
        hidden_df_len = []
        for each in range(1, dest - 1):
            hidden_df_len.append(len(df[source_layer == each]['Weight']))
        weight_file_output = (input_df_len, hidden_df_len, output_df_len)

        if network_file_output == weight_file_output:
            return True
        else:
            temp =[]
            weight_file = (weight_file_output[0],weight_file_output[2])
            network_file = (network_file_output[0],network_file_output[2])
            result = abs(np.subtract(weight_file,network_file))
            hidden_layer_weight = weight_file_output[1]
            hidden_layer_network = network_file_output[1]
            for items in izip_longest(hidden_layer_weight,hidden_layer_network):
                items = np.array(items, dtype=float)
                temp.append(items)
                hidden_diff = abs(np.diff(temp))

            raise ValidationError("Weight Matrix Count Validation Failed \nThe Computation from Weights" \
                                  " File didn\'t match the records in NetworkStructure File.\nThe Number of connection"
                                  " Weights missed layerwise is {0}.\n\nThe Second element which is a list shows an"
                                  " array with each hidden layer and a 'nan' if present, signifies layer "
                                  " mismatch between network structure file and weight file.\nIgnore the floating values"
                                  " in the hidden layer difference as its a NaN float array;treat it as whole number "
                                  " file".format((result[0],hidden_diff.tolist(),result[1])))

    def file_format_validate(self):
        """
        Format Validation wrt weights.txt files
        :return: True if validation is Successful
        """
        networkpath = os.path.abspath('../config/weights.txt')
        header = ['Source Layer', 'Source Node', 'Destination Layer', 'Destination Node', 'Weight']
        df = pd.read_csv(networkpath, delimiter='\t', names=header)
        weight_df = pd.read_csv(self.full_path)

        # Check of Dimension
        bools = (df.shape[1] == weight_df.shape[1])
        if not bools:
            raise ValidationError(""
                                  "Validation Failed."
                                  "\nShape Mismatch of weights file "
                                  "with weights.txt file during File Format validation")
        # Weights are Numeric
        df_format = df['Weight'].dtypes
        bool_num = df_format == weight_df['Weight'].dtypes == 'float64'
        if not bool_num:
            raise TypeError(""
                            "Validation Failed."
                            "\nCheck the Weight file for data type mismatch")

        # position of Weight Vector
        pos = df.columns.get_loc("Weight") == weight_df.columns.get_loc("Weight")
        if not pos:
            raise ValidationError(""
                                  "Validation Failed."
                                  "\nPosition Mismatch found during File Format Validation")

        if pos and bools and bool_num:
            return True

    def redundancy_check(self):
        """

        Returns: False if there are no redundant rows in weights file

        """
        df = pd.read_csv(self.full_path)
        result = df.duplicated(['Source Layer', 'Source Node', 'Destination Layer', 'Destination Node'])
        result = result.tolist()
        if any(result):
            raise ValidationError("Validation Failed.\n"
                                  "Redundancy Validation Failed at Row Index{0}"
                                  " of the generated Weights file".format(np.where(result)[0]))
        else:
            return False

    def validate(self):
        """

        Returns: True If all the Validation is Success.Acts like a driver
        Usage: from validate_weights import WeightValidate
            weights = WeightValidate('C:\\ANN\\ANN_Source\\Data', 'weights.csv')
            if weights.validate():
                    Handle the Boolean Here"
        """

        if self.file_format_validate() and not (self.redundancy_check())and self.weightfile_nan() \
                and self.weight_count_validate():
            return True
