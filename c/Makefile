# Link to Homebrew's compiler because OpenMP support is ...not on Mac...

# Test if default clang supports -fopenmp
OPENMP_SUPPORTED := $(shell echo "int main() { return 0; }" | clang -x c -fopenmp -o /dev/null - 2>/dev/null && echo "yes" || echo "no")

# If default clang does not support -fopenmp, find Homebrew LLVM version
ifeq ($(OPENMP_SUPPORTED),yes)
    CC := clang
	INC := $(CONDA_PREFIX)/include
else
    LLVM_PREFIX := $(shell ls -d /opt/homebrew/Cellar/llvm/* 2>/dev/null | sort -V | tail -n 1)
    ifneq ($(LLVM_PREFIX),)
        CC := $(LLVM_PREFIX)/bin/clang
    else
        $(error No suitable clang found. Install LLVM via Homebrew: brew install llvm libomp)
    endif
	INC := /opt/homebrew/include
endif

# Link to OpenMP library in the user's Conda distribution
LIB := $(CONDA_PREFIX)/lib/

# Actual options here.
default:
	$(CC) dot.cpp -o dot.so -O3 -fPIC -shared -I$(INC) -L$(LIB) -fopenmp -lomp -Wl,-rpath,$(LIB)
clean:
	rm *.so