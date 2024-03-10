NAME        := c_ml

SRCS        := main.c
OBJS        := main.o

CC          := gcc
CFLAGS      := -Wall -Wextra -Werror
LDFLAGS     := -lm

RM          := rm -f
MAKEFLAGS   += --no-print-directory

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(NAME)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)

re:
	$(MAKE) fclean
	$(MAKE) all

mem_check:
	valgrind --leak-check=full \
		 --show-leak-kinds=all \
		 --track-origins=yes \
		 --verbose \
		 ./$(NAME)

.PHONY: clean fclean re
