src=$(shell find ./src -name "*.c")
objects=$(addsuffix .o, $(addprefix $(objectsDir)/,$(shell basename -a $(src) 2>/dev/null | awk -F . '{print $$1}')))
headers=$(shell find ./src -name "*.h")
objectsDir=./obj
srcDir=./src
binDir=./bin
CC=gcc

$(objects): $(objectsDir)/%.o: $(srcDir)/%.c $(headers)
	mkdir -p $(objectsDir)
	$(CC) -Wall -Wextra -DLOG_LEVEL=3 -g -c $< -o $@

$(binDir)/debug: $(objects)
	mkdir -p $(binDir)
	$(CC) -lm $^ -o $@

$(binDir)/release: $(src) $(headers)
	mkdir -p $(binDir)
	$(CC) -DLOG_LEVEL=1 -O3 $^ -o $@

all: debug