CC      = cc
CFLAGS  = -std=c99 -O3 -Wall -g
LDFLAGS = -lm

all: cg cg-sec7 cg-sec8

cg: cg.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

cg-sec7: cg-sec7.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

cg-sec8: cg-sec8.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f cg

.PHONY: clean
