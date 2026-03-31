CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -g
LDFLAGS = -shared -fPIC

SRC     = src/page_allocator.c src/kv_trie.c src/view_builder.c src/march.c
OBJ     = $(SRC:.c=.o)
LIB     = libmarch.so

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

src/%.o: src/%.c
	$(CC) $(CFLAGS) -fPIC -Isrc -c -o $@ $<

clean:
	rm -f src/*.o $(LIB)

