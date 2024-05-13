#!/bin/bash

# Build script for GitHub Actions

# Setup gRPC
export MY_INSTALL_DIR=$HOME/.local
mkdir -p $MY_INSTALL_DIR
export PATH="$MY_INSTALL_DIR/bin:$PATH"
git clone --recurse-submodules -b v1.58.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc

cd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
	-DgRPC_BUILD_TESTS=OFF \
	-DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
	../..
make -j
make install
popd

# Setup ONNXRuntime
cd $HOME
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz

# Setup MLCompilerBridge
REPO_DIR=$GITHUB_WORKSPACE
BUILD=$1

if [[ -z "$BUILD" ]]; then
	echo "Pls provide build type"
	echo "exiting..."
	exit 1
fi

cmake \
	-G "Unix Makefiles" \
	-S $REPO_DIR \
	-B $REPO_DIR/build_${BUILD,,} \
	-DONNXRUNTIME_ROOTDIR=$HOME/onnxruntime-linux-x64-1.16.3 \
	-DTENSORFLOW_AOT_PATH=$CONDA/envs/mlbridge/lib/python3.10/site-packages/tensorflow \
	-DCMAKE_BUILD_TYPE="$BUILD" \
	-DCMAKE_INSTALL_PREFIX=$REPO_DIR/install \
	-DPROTOS_DIRECTORY=$REPO_DIR/test/protos \
	-DPYTHON_UTILITIES_DIRECTORY=$REPO_DIR/test \
	-DMLBRIDGE_ENABLE_TEST=ON

make -j -C $REPO_DIR/build_${BUILD,,} install
