set -euxo pipefail

# Install Dependencies
cd $WORKING_DIR
pip install -e ".[vllm_latest]"
pip install pebble
pip install https://github.com/tongyx361/symeval.git

nvcc --version
pip list -v