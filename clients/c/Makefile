CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c99

examples/heat2d: examples/heat2d.c numbl_plot.h
	$(CC) $(CFLAGS) -I. $< -lm -pthread -o $@

clean:
	rm -f examples/heat2d

.PHONY: clean
