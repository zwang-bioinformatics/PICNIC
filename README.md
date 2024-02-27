# PICNIC2-LOCAL and PICNIC2-GLOBAL README

## Introduction
This README provides instructions for using PICNIC2-LOCAL and PICNIC2-GLOBAL, two 3D ResNets designed to refine AlphaFold protein tertiary structures. These tools are aimed at enhancing the accuracy of protein structure predictions by leveraging machine learning techniques.

## PICNIC2-LOCAL

### Description
PICNIC2-LOCAL refines protein tertiary structure one atom at a time. It will only run on the CASP15 input models in the `local/casp15_af_models/` directory, as it uses MASS2 and LAW features, which are already generated for those models.

### Usage
To refine a protein structure using PICNIC2-LOCAL, follow these steps:
1. Ensure that you use a given input model from a CASP15 target in the `local/casp15_af_models/` directory.
2. Run the refinement script using the following command:
   ```
   python run_PICNIC2-LOCAL.py local/casp15_af_models/{target_name}/{model_name}
   ```
   Replace `local/casp15_af_models/{target_name}/{model_name}` with the path to the specific AlphaFold model you want to refine. An example model is located at `local/casp15_af_models/T1104/af2-standard_T1104_1`
3. The refined structure will be generated and saved in the `local/out_casp15/` directory.

## PICNIC2-GLOBAL

### Description
PICNIC2-GLOBAL refines the entire protein tertiary structure at once. Unlike PICNIC2-LOCAL, it does not have specific requirements for input model.

### Usage
To refine a protein structure using PICNIC2-GLOBAL, follow these steps:
1. Ensure that you have the protein tertiary structure file (in PDB format) ready. You can use the provided example file named `sample.pdb`.
2. Run the refinement script using the following command:
   ```
   python run_PICNIC2-GLOBAL.py {path_to_pdb}
   ```
   Replace `{path_to_pdb}` with the path to your input protein structure file. For an example, use `sample.pdb`
3. The refined structure will be generated and saved in the same directory as the input file.

## Note
Both PICNIC2-LOCAL and PICNIC2-GLOBAL utilize advanced machine learning techniques to refine protein structures. It's essential to have the necessary dependencies and libraries installed before running the refinement scripts. Additionally, please ensure that you have sufficient computational resources available to perform the refinement process effectively. For more information on the dependencies and usage, please refer to the documentation provided with the tools.