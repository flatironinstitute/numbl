CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c99
LDLIBS = -lm -pthread

all: libnumbl_plot.a examples/heat2d

numbl_plot.o: numbl_plot.c numbl_plot.h
	$(CC) $(CFLAGS) -pthread -c numbl_plot.c -o $@

libnumbl_plot.a: numbl_plot.o
	$(AR) rcs $@ $^

examples/heat2d: examples/heat2d.c libnumbl_plot.a
	$(CC) $(CFLAGS) -I. $< -L. -lnumbl_plot $(LDLIBS) -o $@

clean:
	rm -f numbl_plot.o libnumbl_plot.a examples/heat2d

.PHONY: all clean
