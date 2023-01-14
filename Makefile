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

# compila tutti i file .c e .h in src e sue sottocartelle in files .o in obj ottimizzando le informazioni di debug.
# In questa maniera sono compilati solo i sorgenti che sono stati modificati, velocizzando la build.
$(objects): $(objectsDir)/%.o: $(srcDir)/%.c $(headers)
	mkdir -p $(objectsDir) $(objectsSubDirs)
	$(CC) $(INCLUDES) -Wall -Wextra -DLOG_LEVEL=3 -g -c $< -o $@

# linka i file .o in obj e crea il file debug
$(binDir)/debug: $(objects)
	mkdir -p $(binDir)
	$(CC) -lm $^ -o $@

# compila tutti i sorgenti in src in un eseguibile ottimizzato per le prestazioni
$(binDir)/release: $(src) $(headers)
	mkdir -p $(binDir)
	$(CC) $(INCLUDES) -DLOG_LEVEL=1 -O3 $^ -o $@

all: debug