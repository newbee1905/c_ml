NAME = c_ml

CC = gcc
CFLAGS = -Wall -Wextra -Iinclude
LDFLAGS = -Llib
LDLIBS = -l$(NAME) -lm

SRCDIR = src
INCDIR = include
BINDIR = bin
OBJDIR = obj
LIBDIR = lib

DEMO_SRC = main.c
SRCS := $(wildcard $(SRCDIR)/*.c)
OBJS := $(filter-out $(OBJDIR)/main.o, $(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o))
LIB = $(LIBDIR)/lib$(NAME).a
EXECUTABLE = $(NAME)

.PHONY: all clean library

all: $(BINDIR)/$(EXECUTABLE)

$(BINDIR)/$(EXECUTABLE): $(OBJS) $(LIB) | $(BINDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(DEMO_SRC) $^ -o $@ $(LDLIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJS) | $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

$(LIBDIR):
	mkdir -p $(LIBDIR)

clean:
	$(RM) -r $(BINDIR) $(OBJDIR) $(LIB)

mem_check:
	valgrind --leak-check=full \
		 --show-leak-kinds=all \
		 --track-origins=yes \
		 --verbose \
		 $(BINDIR)/$(EXECUTABLE)

library: $(LIB)

