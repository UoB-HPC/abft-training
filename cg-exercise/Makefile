CC      = cc
CFLAGS  = -std=c99 -O3 -Wall -g
LDFLAGS = -lm

EXES = cg cg-sec7 cg-sec8

all: $(EXES)

cg: cg.c ecc.c spmv.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

cg-sec7: cg.c ecc.c spmv-sec7.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

cg-sec8: cg.c ecc.c spmv-sec8.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(EXES)

.PHONY: clean
