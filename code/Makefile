CC      = cc
CFLAGS  = -std=c99 -O3 -Wall
LDFLAGS = -lm

all: cg

cg: cg.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f cg
