DIFFSOL_C_DIR ?= ../diffsol/target/release

.PHONY: build
build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
		-DDIFFSOL_C_DIR=$(DIFFSOL_C_DIR) \
		-DXLA_FFI_INCLUDE_DIR="$$(.venv/bin/python -c 'import jax.ffi; print(jax.ffi.include_dir())')"
	cmake --build build -j4

.PHONY: install
install:
	CMAKE_ARGS="-DDIFFSOL_C_DIR=$(DIFFSOL_C_DIR)" uv pip install -e .

.PHONY: test
test:
	.venv/bin/python -m pytest tests -q

.PHONY: clean
clean:
	rm -rf build
