CC=gcc
CXX=g++-4.6
CFLAGS=
NVCC=nvcc -arch=sm_21 -w

INCLUDE= -I include/ \
	 -I /usr/local/boton/include/ \
 	 -isystem /usr/local/cuda/samples/common/inc/ \
	 -isystem /usr/local/cuda/include

CPPFLAGS= -std=c++0x -Werror -Wall $(CFLAGS) $(INCLUDE)

SOURCES=utility.cpp dnn.cpp

EXECUTABLES=
EXAMPLE_PROGRAM=example
 
.PHONY: debug all o3 example ctags
all: $(EXECUTABLES) $(EXAMPLE_PROGRAM) ctags

o3: CFLAGS+=-O3
o3: all
debug: CFLAGS+=-g -DDEBUG
debug: all

vpath %.h include/
vpath %.cpp src/
vpath %.cu src/

OBJ=$(addprefix obj/,$(SOURCES:.cpp=.o))

LIBRARY= -lmatrix

LIBRARY_PATH=-L/usr/local/boton/lib/

example: $(OBJ) example.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)

# +==============================+
# +===== Other Phony Target =====+
# +==============================+
obj/%.o: %.cpp
	$(CXX) $(CPPFLAGS) -o $@ -c $<

obj/%.o: %.cu
	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

obj/%.d: %.cpp
	@$(CXX) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,obj/\1.o $@ : ,g' < $@.$$$$ > $@;\
	rm -f $@.$$$$

-include $(addprefix obj/,$(subst .cpp,.d,$(SOURCES)))

.PHONY: ctags
ctags:
	@ctags -R *
clean:
	rm -rf $(EXECUTABLES) $(EXAMPLE_PROGRAM) obj/*
