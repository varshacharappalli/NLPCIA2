CC = gcc
CFLAGS = -Wall -O2

ifeq ($(OS),Windows_NT)
    TARGET = preprocess.exe
    RM = del /Q
else
    TARGET = preprocess
    RM = rm -f
endif

all: $(TARGET)

$(TARGET): preprocess.c
	$(CC) $(CFLAGS) -o $(TARGET) preprocess.c

clean:
	$(RM) $(TARGET)
