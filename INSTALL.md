# PUNCTUATION

## Create CONDA environment

```
# Create conda environment
unset PYTHONHOME && \
conda create --prefix /mnt/matylda5/iveselyk/CONDA_ENVS/punctuation python=3.9

# activate conda env
unset PYTHONHOME && \
conda activate /mnt/matylda5/iveselyk/CONDA_ENVS/punctuation

### install 'debug' tools:
python -m pip install ipython matplotlib

### install pytorch:
python -m pip install torch # 900MB, torch-1.13.1-cp39-cp39-manylinux1_x86_64.whl

### install other dependencies:
git clone https://github.com/EU-ASR/punctuation-restoration; cd punctuation-restoration
python -m pip install -r requirements.txt

# note: I had to increase the version of transformers: v2.11.0 -> v4.27.3
#       It seems to be okay, the models are training normally.

```


