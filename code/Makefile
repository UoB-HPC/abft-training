CC      = cc
CFLAGS  = -std=c99 -O3 -Wall -g
LDFLAGS = -lm

all: cg

cg: cg.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f cg
