CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra -pedantic -Iinclude

SRC = src/main.cpp src/data.cpp src/tree.cpp src/experiment.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = risk_aware_shift_trees

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)