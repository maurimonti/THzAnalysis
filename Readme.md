This package provides a simple tool to compute terahertz conductivities and transmissions and a basic fitting routine based on the lmfit package.

THzAnalysis.py contains the calss of THz solver, all the relevant quantites are attributes of the THzSolver object.
The software is ablo to compute spectra and provide an average, if they are part of the same experiment or to consider the different files give as different experiments. This is controlled by the multiplot attribute.

The sample models and fitting models are in ConductivityModels.py
some general functions are in data_manipulation_functions.py and math_functions.py.

There are three file format for the data, these are 'Ox', 'TW' and'abcd':
-'abcd' correspoonds to the picoscope acquision format which assumes the following:
        * x(mm), Eon, Eoff, Eon-Eoff, Eon-Eoff/Eoff, A, B, C, D
-'Ox' correspond to the format used in Oxford, assumes x is col 0, signal is col 2 and ref is col 1
- 'TW' is the format correspondant to the Teraview system in Warwick format: assumes signal is col1, x is col 0 in optical delay and reference is col 7.
if none of the above is specified it assumes the signal and referene columns are the col 1 of the corresponding files and x is the col 0.

To write a new sample open sample.py and copy the format of the sample class, default values are indicated there.
Again the models allowed are in the file ConductivityModels.py, to write another one just follow the template and change the SwitchTemp method as appropriate.

