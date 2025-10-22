#ifndef TYPES_HPP
#define TYPES_HPP

#if MIXED_PRECISION
    using CL_TYPE = float;
#else
    using CL_TYPE = double;
#endif

#endif