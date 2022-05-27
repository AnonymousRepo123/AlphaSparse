#ifndef UTILITIES_H
#define UTILITIES_H

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <cuda.h>

using namespace std;

//Generates the dense vector required for multiplication
float *vect_gen(int n, bool isSerial = false)
{
    float *vect = new float[n];

    for (int i = 0; i < n; i++)
    {
        if (!isSerial)
            vect[i] = rand() % 10;
        else
            vect[i] = i + 1;
    }

    return vect;
}

#endif