CC      = gcc
CFLAGS  = -std=c99 -Wall -O3 -ffast-math
LDFLAGS = -lm

PLATFORM = $(shell uname -s)
ifneq ($(PLATFORM), Darwin)
	CFLAGS += -fopenmp
endif

all: miniBUDE

miniBUDE: miniBUDE.c crc32.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	echo "	$(CC) $(CFLAGS) $^ $(LDFLAGS)"

clean:
	rm -f miniBUDE
