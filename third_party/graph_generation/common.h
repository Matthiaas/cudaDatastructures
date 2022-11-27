#ifndef TEAM02_COMMON_H
#define TEAM02_COMMON_H

#include <cinttypes>

typedef uint32_t index_t;
#define INDEX_FMT "%" PRIu32

static inline index_t roundUp(index_t numToRound, index_t multiple) 
{
    return (numToRound + multiple - 1) & -multiple;
}


#endif //TEAM02_COMMON_H
