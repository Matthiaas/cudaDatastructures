#!/bin/bash

./build/graph_generator -gt GERMAN_ROAD_NETWORK -o generated_graphs/GERMAN_ROAD_NETWORK.txt
./build/graph_generator -gt COMP_SCIENCE_AUTHORS -o generated_graphs/COMP_SCIENCE_AUTHORS.txt
./build/graph_generator -gt GOOGLE_CONTEST -o generated_graphs/GOOGLE_CONTEST.txt
./build/graph_generator -gt HEP_LITERATURE -o generated_graphs/HEP_LITERATURE.txt
./build/graph_generator -gt WWW_NOTRE_DAME -o generated_graphs/WWW_NOTRE_DAME.txt
./build/graph_generator -gt US_PATENTS -o generated_graphs/US_PATENTS.txt
