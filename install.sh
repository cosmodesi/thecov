set -e  # Exit on error

# Detect the operating system (they require slightly different dependencies)
OS=$(uname -s)
ARCH=$(uname -m)

echo "Operating system: $OS"
echo "Architecture: $ARCH"

if [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
    PLATFORM=linux-64
    export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu/ # <- needed to build mpi4py
elif [[ "$OS" == "Darwin" ]]; then
    PLATFORM=osx-arm64
    # NOTE: You need to use the homebrew version of clang-15 for this to work
    # https://stackoverflow.com/questions/71061894/how-to-install-openmp-on-mac-m1
else
    echo "ERROR: Unsupported platform detected! Installation failed!"
    exit 1
fi

########### conda stuff
# Always execute this script with bash, so that conda shell.hook works.
# Relevant conda bug: https://github.com/conda/conda/issues/7980
if [[ -z "$BASH_VERSION" ]];
then
    exec bash "$0" "$@"
fi

eval "$(conda shell.bash hook)"

echo "Creating new anaconda enviornment..."
conda create -n thecov python=3.11 --platform $PLATFORM
conda activate thecov

# Run pip install for dependencies in pyproject.toml
echo "installing some common packages first..."
pip install six==1.15.0 numpy==1.26 scipy Cython h5py mpi4py jupyter

# finally, thecov itself
echo "Done! installing thecov..."
python -m pip install -e .

echo "Done! activate the new enviornment with 'conda activate thecov'"