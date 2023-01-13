src=$(shell find ./src -name "*.c")
srcSubDirs=$(shell find ./src -type d)
objects=$(src:./src/%.c=./obj/%.o)
headers=$(shell find ./src -name "*.h")
objectsDir=./obj
objectsSubDirs=$(srcSubDirs:./src/%=./obj/%)
srcDir=./src
binDir=./bin
CC=gcc
INCLUDES=-I./src

$(objects): $(objectsDir)/%.o: $(srcDir)/%.c $(headers)
	mkdir -p $(objectsDir) $(objectsSubDirs)
	$(CC) $(INCLUDES) -Wall -Wextra -DLOG_LEVEL=3 -g -c $< -o $@

$(binDir)/debug: $(objects)
	mkdir -p $(binDir)
	$(CC) -lm $^ -o $@

$(binDir)/release: $(src) $(headers)
	mkdir -p $(binDir)
	$(CC) $(INCLUDES) -DLOG_LEVEL=1 -O3 $^ -o $@

all: debug