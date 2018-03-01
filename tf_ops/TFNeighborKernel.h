//
// Created by pal on 18-2-8.
//

#ifndef POINTUTIL_TFNEIGHBORKERNEL_H
#define POINTUTIL_TFNEIGHBORKERNEL_H


template<typename FLT_TYPE,typename INT_TYPE>
void neighborScatterGPU(
        FLT_TYPE *d_ifeats,          // [pn,ifn]
        INT_TYPE *d_inidxs,      // [pn,n] or [csum]
        INT_TYPE *d_inidxs_lens, // [pn] n=nidxs_lens[i]
        INT_TYPE *d_inn_bgs,     // [pn] n=nidxs_lens[i]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *d_sfeats,           // [csum,ifn]
        bool use_diff                 // default false
);

template<typename FLT_TYPE,typename INT_TYPE>
void neighborGatherGPU(
        FLT_TYPE *d_sfeats,           // [pn,ifn]
        INT_TYPE *d_inidxs,       // [pn,n] or [csum]
        INT_TYPE *d_inidxs_lens,  // [pn] n=nidxs_lens[i]
        INT_TYPE *d_inn_bgs,      // [pn] n=nidxs_lens[i]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *d_ifeats,           // [csum,ifn]
        bool use_diff                 // default false
);

template<typename FLT_TYPE,typename INT_TYPE>
void locWFeatSumForwardGPU(
        FLT_TYPE *d_itfeats,              // [pn,n,m,ofn]
        FLT_TYPE *d_ilw,                  // [pn,n,m]
        INT_TYPE *d_inidxs_lens,      // [pn] inidxs_lens[i]=n
        INT_TYPE *d_inn_bgs,          // [pn] inidxs_lens[i]=n
        INT_TYPE pn,
        INT_TYPE ofn,
        INT_TYPE m,
        FLT_TYPE *d_otfeats_sum           // [pn,m,ofn]
);

template<typename FLT_TYPE,typename INT_TYPE>
void locWFeatSumBackwardGPU(
        FLT_TYPE *d_itfeats,              // [pn,n,m,ofn]
        FLT_TYPE *d_ilw,                  // [pn,n,m]
        FLT_TYPE *d_dotfeats_sum,         // [pn,m,ofn]
        INT_TYPE *d_inidxs_lens,      // [pn] inidxs_lens[i]=n
        INT_TYPE *d_inn_bgs,          // [pn] inidxs_lens[i]=n
        INT_TYPE csum,
        INT_TYPE pn,
        INT_TYPE ofn,
        INT_TYPE m,
        FLT_TYPE *d_ditfeats,           // [pn,n,m,ofn]
        FLT_TYPE *d_dilw                // [pn,n,m]
);


template<typename FLT_TYPE,typename INT_TYPE>
void locWSumForwardGPU(
        FLT_TYPE *d_ilw,                      // pn,n,m
        INT_TYPE *d_inidxs_lens,          // [pn] n=nidxs_lens[i]
        INT_TYPE *d_inn_bgs,          // [pn] n=nidxs_lens[i]
        INT_TYPE pn,
        INT_TYPE m,
        FLT_TYPE *d_olw_sum                   // pn,m
);

template<typename FLT_TYPE,typename INT_TYPE>
void locWSumBackwardGPU(
        FLT_TYPE *d_dolw_sum,         // pn,m
        INT_TYPE *d_inidxs_lens,      // [pn] n=nidxs_lens[i]
        INT_TYPE *d_inn_bgs,          // [pn] n=nidxs_lens[i]
        INT_TYPE csum,
        INT_TYPE pn,
        INT_TYPE m,
        FLT_TYPE *d_dilw              // [pn,n,m]
);



template<typename FLT_TYPE,typename INT_TYPE>
void neighborSumFeatGatherGPU(
        FLT_TYPE *d_ifeats,               // [csum,fd]
        INT_TYPE *d_icidxs,               // [csum]
        INT_TYPE pn,
        INT_TYPE fd,
        INT_TYPE csum,
        FLT_TYPE *d_ogfeats_sum           // [pn,fd]
);

template<typename FLT_TYPE,typename INT_TYPE>
void neighborSumFeatScatterGPU(
        FLT_TYPE *d_igfeats_sum,            // [pn,fd]
        INT_TYPE *d_icidxs,                 // [csum] inidxs_lens[i]=n
        INT_TYPE pn,
        INT_TYPE fd,
        INT_TYPE csum,
        FLT_TYPE *d_osfeats                 // [csum,fd]
);


#endif //POINTUTIL_TFNEIGHBORKERNEL_H
