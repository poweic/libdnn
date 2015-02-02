CXX=g++
CFLAGS=-fPIC -std=c++0x #-Werror -Wall
LFLAGS_BIN=
LFLAGS_LIB=-shared
NVCC=nvcc -arch=sm_21 -w #-Xcompiler "-Wall"
NVCFLAGS=-Xcompiler -fPIC

BOTON_UTIL_ROOT=tools/utility/
CUMATRIX_ROOT=tools/libcumatrix/

INCLUDE= -I ./ \
	 -I include/ \
	 -I $(BOTON_UTIL_ROOT)/include/ \
	 -I $(CUMATRIX_ROOT)/include \
 	 -I /usr/local/cuda/samples/common/inc/ \
	 -I /usr/local/cuda/include

SOURCES=cnn-utility.cu\
	nnet.cpp\
	dnn-utility.cu\
	utility.cpp\
	rbm.cu\
	feature-transform.cu\
	data-io.cpp\
	dataset.cpp\
	batch.cpp\
	config.cpp

EXECUTABLES=nn-train\
	    nn-predict\
	    nn-init\
	    nn-info\
	    nn-print\
	    data-statistics

LIBRARY=libdnn.so


EXECUTABLES:=$(addprefix bin/, $(EXECUTABLES))
LIBRARY:=$(addprefix bin/, $(LIBRARY))

.PHONY: debug all o3 dump_nrv ctags clean

all: $(EXECUTABLES) $(LIBRARY) ctags

o3: CFLAGS+=-O3
o3: all
debug: CFLAGS+=-g -DDEBUG
debug: all
dump_nrv: NVCC+=-Xcompiler "-fdump-tree-nrv" all
dump_nrv: all

vpath %.h include/
vpath %.cpp src/
vpath %.cu src/

OBJDIR:=obj/
OBJ:=$(addprefix $(OBJDIR)/, $(addsuffix .o,$(basename $(SOURCES))))

$(OBJDIR):
	@mkdir -p $(OBJDIR)

UTIL_LIBS=-lpbar -lcumatrix
CUDA_LIBS=-lcuda -lcudart -lcublas
LIBRARY_PATH=-L$(BOTON_UTIL_ROOT)/lib/ -L$(CUMATRIX_ROOT)/lib -L/usr/local/cuda/lib64

$(EXECUTABLES): bin/% : $(OBJ) $(OBJDIR)/%.o
	$(CXX) -o $@ $(LFLAGS_BIN) $^ $(LIBRARY_PATH) $(UTIL_LIBS) $(CUDA_LIBS)

$(LIBRARY): bin/% : $(OBJ)
	$(CXX) -o $@ $(LFLAGS_LIB) $^ $(LIBRARY_PATH)

# +==============================+
# +===== Other Phony Target =====+
# +==============================+
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ -c $<

$(OBJDIR)/%.o: %.cu include/%.h
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o $@ -c $<

$(OBJDIR)/%.d: %.cpp | $(OBJDIR)
	@$(CXX) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,$(OBJDIR)/\1.o $@ : ,g' < $@.$$$$ > $@;\
	rm -f $@.$$$$

-include $(addprefix $(OBJDIR)/,$(subst .cpp,.d,$(SOURCES)))

ctags:
	@if command -v ctags >/dev/null 2>&1; then ctags -R --langmap=C:+.cu *; fi
clean:
	rm -rf $(EXECUTABLES) $(OBJDIR)/*
