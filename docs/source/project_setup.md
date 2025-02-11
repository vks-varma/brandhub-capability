# Environment Setup Guide

This guide provides step-by-step instructions to set up a Conda environment, install dependencies from `req.txt`, configure necessary settings, and run the `main.py` script.

## 1. **Create a Conda Environment**
First, create a new Conda environment with Python 3.10 (or your preferred version):

```sh
conda create --name my_project_env python=3.10 -y
```

Activate the environment:

```sh
conda activate my_project_env
```

## 2. **Install Dependencies**
Install all required dependencies from `req.txt`:

```sh
pip install -r req.txt
```

Ensure all dependencies are installed correctly:

```sh
pip list
```

## 3. **Make Necessary Config Changes**
Edit the `config.yml` file located in `src/BHC_Capability/` if needed. Open the file and update any required settings:

```yaml
# Example config.yml
param1: value1
param2: value2
```

You can use a text editor to modify it:

```sh
nano src/BHC_Capability/config.yml
```

## 4. **Run the Main Script**
Once the environment is set up and the configuration is updated, run the `main.py` script:

```sh
python src/BHC_Capability/main.py
```

If you need to run a specific module, adjust the command accordingly:

```sh
python src/BHC_Capability/score.py
```

## 5. **Deactivate the Environment (Optional)**
After you're done, you can deactivate the Conda environment:

```sh
conda deactivate
```

## **Troubleshooting**
- If `conda` is not recognized, ensure Conda is installed and added to your system path.
- If dependencies fail to install, try updating `pip`:
  ```sh
  pip install --upgrade pip
  ```
- If `main.py` does not run due to missing modules, verify that your `sys.path` includes the correct directories in `conf.py`.

---
This setup ensures a smooth workflow for running your project with all necessary dependencies and configurations.

