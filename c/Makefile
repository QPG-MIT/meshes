# Link to Homebrew's compiler because OpenMP support is ...not on Mac...
CC := /opt/homebrew/Cellar/llvm/15.0.7_1/bin/clang
INC := /opt/homebrew/include
# Link to OpenMP library in the user's Conda distribution
LIB := $(CONDA_PREFIX)/lib/

default:
	$(CC) dot.cpp -o dot.so -O3 -fPIC -shared -I$(INC) -L$(LIB) -fopenmp -Wl,-rpath,$(LIB)
clean:
	rm *.so