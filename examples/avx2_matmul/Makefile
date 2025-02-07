CFLAGS ?= -march=native

.PHONY: x86
x86: avx2_matmul

# x86 build
avx2_matmul: avx2_matmul.o main.o
avx2_matmul.h avx2_matmul.c: x86_matmul.py
	exocc -o . --stem $(*F) $^

.PHONY: neon
neon: neon_matmul

# ARM 
neon_matmul: neon_matmul.o main.o
neon_matmul.h neon_matmul.c: arm_matmul.py
	exocc -o . --stem $(*F) $^

.PHONY: clean
clean:
	$(RM) *.o exo_demo
	$(RM) -r __pycache__/
	$(RM) avx2_matmul avx2_matmul.* 
	$(RM) neon_matmul neon_matmul.*
	
