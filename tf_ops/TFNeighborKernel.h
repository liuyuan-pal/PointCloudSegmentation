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
        INT_TYPE *d_nidxs_lens,           // [pn]
        INT_TYPE *d_nidxs_bgs,            // [pn]
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

template<typename FLT_TYPE,typename INT_TYPE>
void neighborMaxFeatGatherGPU(
        FLT_TYPE *d_ifeats,               // [pn1,fd]
        INT_TYPE *d_vlens,                // [pn2]
        INT_TYPE *d_vlens_bgs,            // [pn2]
        INT_TYPE pn2,
        INT_TYPE fd,
        FLT_TYPE *d_ogfeats_sum,          // [pn2,fd]
        INT_TYPE *d_ogmax_idxs            // [pn2,fd]
);

template<typename FLT_TYPE,typename INT_TYPE>
void neighborMaxFeatScatterGPU(
        FLT_TYPE *d_igfeats_sum,            // [pn2,fd]
        INT_TYPE *d_igmax_idxs,             // [pn2,fd]
        INT_TYPE *vlens_bg,                 // [pn2]
        INT_TYPE *vlens,                    // [pn2]
        INT_TYPE pn1,
        INT_TYPE pn2,
        INT_TYPE fd,
        FLT_TYPE *d_osfeats                 // [pn1,fd]
);

template<typename FLT_TYPE,typename INT_TYPE>
void concatNonCenterFeatGatherGPU(
        FLT_TYPE *d_sfeats,                  // [pn*(n-1),2*ifn]
        INT_TYPE *d_inidxs,                  // [n]
        INT_TYPE *d_inidxs_lens,             // [pn]
        INT_TYPE *d_inn_bgs,                 // [pn]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *d_ifeats                   // [pn,ifn]
);

template<typename FLT_TYPE,typename INT_TYPE>
void concatNonCenterFeatScatterGPU(
        FLT_TYPE *d_ifeats,                   // [pn,ifn]
        INT_TYPE *d_inidxs,                   // [pn,n]
        INT_TYPE *d_inidxs_lens,              // [pn]
        INT_TYPE *d_inn_bgs,                  // [pn]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *d_onfeats                   // [pn*(n-1),2*ifn]
);

template<typename INT_TYPE>
void eliminateCenterGPU(
        INT_TYPE *inidxs,                   // [n]
        INT_TYPE *inidxs_lens,              // [pn]
        INT_TYPE *inidxs_bgs,               // [pn]
        INT_TYPE pn,
        INT_TYPE *onidxs,                   // [n-pn]
        INT_TYPE *onidxs_lens,              // [pn]
        INT_TYPE *onidxs_bgs,               // [pn]
        INT_TYPE *ocidxs                    // [n-pn]

);

#endif //POINTUTIL_TFNEIGHBORKERNEL_H
