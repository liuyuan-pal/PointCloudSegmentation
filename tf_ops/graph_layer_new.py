from graph_pooling_layer import *


def diff_feats_ecd(sxyzs, feats, ifn, ifc_dims, ofc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name, reuse=None):
    with tf.name_scope(name):
        sfeats = neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=True)  # [en,ifn]
        sfeats = tf.concat([sfeats,sxyzs],axis=1)

        for idx,fd in enumerate(ifc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=ifn, scope='{}_fc_ew'.format(name),
                                             activation_fn=tf.nn.tanh, reuse=reuse)

        feats=neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=False)      # [en,ifn]
        feats=ew*feats
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps)/(tf.cast(nidxs_lens,tf.float32)+eps),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=tf.nn.relu, reuse=reuse)

        return feats


def diff_feats_ecd_v2(sxyzs, feats, ifn, ifc_dims, ofc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name,
                      use_l2_norm=None,weight_activation=None,final_activation=None,reuse=None):
    with tf.name_scope(name):
        sfeats = neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=True)  # [en,ifn]
        sfeats = tf.concat([sfeats,sxyzs],axis=1)

        for idx,fd in enumerate(ifc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=ifn, scope='{}_fc_ew'.format(name),
                                             activation_fn=weight_activation, reuse=reuse)

        if use_l2_norm:
            norm=tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(ew),axis=1)+1e-5),axis=1)
            ew/=(norm+1e-5)
            with tf.variable_scope(name):
                weights_transformer=variable_on_cpu('edge_weights_trans',[1,ifn],tf.ones_initializer)
                ew*=weights_transformer

        feats=neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=False)      # [en,ifn]
        feats=ew*feats
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps)/(tf.cast(nidxs_lens,tf.float32)+eps),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=final_activation, reuse=reuse)

        return feats


def diff_feats_ecd_xyz(sxyzs, feats, ifn, ifc_dims, ofc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name, reuse=None):
    with tf.name_scope(name):
        sfeats = sxyzs

        for idx,fd in enumerate(ifc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=ifn, scope='{}_fc_ew'.format(name),
                                             activation_fn=tf.nn.tanh, reuse=reuse)

        feats=neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=False)      # [en,ifn]
        feats=ew*feats
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps)/(tf.cast(nidxs_lens,tf.float32)+eps),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=tf.nn.relu, reuse=reuse)

        return feats


def diff_xyz_ecd(sxyzs, ifn, ifc_dims, ofc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name, reuse=None):
    with tf.name_scope(name):
        sfeats=sxyzs
        dim_sum=3
        for idx,fd in enumerate(ifc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)
            dim_sum+=fd

        ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=dim_sum, scope='{}_fc_ew'.format(name),
                                             activation_fn=tf.nn.tanh, reuse=reuse)

        feats=ew*sfeats                                                                               # [en,ifn]
        # we need to embed the edge-conditioned feature to avoid signal mixed up
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps)/(tf.cast(nidxs_lens,tf.float32)+eps),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]

        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=tf.nn.relu, reuse=reuse)

        return feats


def pointnet_conv(sxyzs, feats, fc_dims, ofn, name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        sfeats = graph_concat_scatter(feats,nidxs,nlens,nbegs,ncens)
        sfeats = tf.concat([sfeats,sxyzs],axis=1)

        for idx,fd in enumerate(fc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=None, reuse=reuse)
        feats=graph_pool(sfeats,nlens,nbegs)
        return feats

def pointnet_conv_v2(sxyzs, feats, fc_dims, ofn, name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        sfeats = graph_concat_scatter(feats,nidxs,nlens,nbegs,ncens)
        sfeats = tf.concat([sfeats,sxyzs],axis=1)

        for idx,fd in enumerate(fc_dims):
            sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)

        sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=None, reuse=reuse)
        feats=graph_pool(sfeats,nlens,nbegs)
        return feats


def pointnet_pool(xyzs, feats, fc_dims, ofn, name, nlens, nbegs, reuse=None):
    with tf.name_scope(name):
        sfeats = tf.concat([xyzs,feats],axis=1)

        for idx,fd in enumerate(fc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                 activation_fn=None, reuse=reuse)
        feats=graph_pool(sfeats,nlens,nbegs)
        return feats


def concat_feats_ecd(sxyzs, feats, ifn, ifc_dims, ofc_dims, ofn, name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        sfeats = graph_concat_scatter(feats,nidxs,nlens,nbegs,ncens)
        sfeats = tf.concat([sfeats,sxyzs],axis=1)

        for idx,fd in enumerate(ifc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=ifn, scope='{}_fc_ew'.format(name),
                                             activation_fn=tf.nn.tanh, reuse=reuse)

        feats=neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)      # [en,ifn]
        feats=ew*feats
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, ncens, nlens, nbegs)  # [pn,ofn]
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=tf.nn.relu, reuse=reuse)

        return feats


def anchor_conv(sxyzs, feats, ifn, ofn, anchor_num, name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)

        with tf.variable_scope(name,reuse=reuse):
            anchor_init = generate_anchor(anchor_num).transpose()
            anchors = variable_on_cpu('anchor', [anchor_num,3], initializer=None, init_val=anchor_init)

        diff=tf.squared_difference(tf.expand_dims(sxyzs,axis=1),tf.expand_dims(anchors,axis=0))
        diff=-tf.reduce_sum(diff,axis=2) # [e,an]
        weights=tf.exp(diff) # [e,an]
        weighted_edge_feats=tf.expand_dims(weights,axis=2)*tf.expand_dims(edge_feats,axis=1) # [e,an,ifn]
        weighted_edge_feats=tf.reshape(weighted_edge_feats,[-1,anchor_num*ifn])

        weighted_point_feats=neighbor_ops.neighbor_sum_feat_gather(weighted_edge_feats,ncens, nlens, nbegs) #[pn,an*ifn]
        output_point_feats = tf.contrib.layers.fully_connected(
            weighted_point_feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
            activation_fn=tf.nn.relu, reuse=reuse)

        return output_point_feats


def anchor_conv_v2(sxyzs, feats, ofn, anchor_num, embed_dim, name,
                   nidxs, nlens, nbegs, ncens,
                   rescale_ratio=4.0, trainable_anchor=True, reuse=None):
    with tf.name_scope(name):
        feats_embed = tf.contrib.layers.fully_connected(
            feats, num_outputs=anchor_num*embed_dim, scope='{}_fc_embed'.format(name),
            activation_fn=None, reuse=reuse)

        # [en,ed*an]
        edge_feats = neighbor_ops.neighbor_scatter(feats_embed, nidxs, nlens, nbegs, use_diff=False)

        with tf.variable_scope(name,reuse=reuse):
            anchor_init = generate_anchor(anchor_num).transpose()
            anchors = variable_on_cpu('anchor', [anchor_num,3], initializer=None,
                                      init_val=anchor_init, trainable=trainable_anchor)

        diff=tf.squared_difference(tf.expand_dims(sxyzs,axis=1),tf.expand_dims(anchors,axis=0))
        diff=-tf.reduce_sum(diff,axis=2) # [en,an]
        diff*=rescale_ratio
        weights=tf.exp(diff) # [en,an]

        # [e,an,en]
        edge_feats=tf.reshape(edge_feats,[-1,anchor_num,embed_dim])
        weighted_edge_feats=tf.expand_dims(weights,axis=2)*edge_feats
        weighted_edge_feats=tf.reshape(weighted_edge_feats,[-1,anchor_num*embed_dim])

        weighted_point_feats=neighbor_ops.neighbor_sum_feat_gather(weighted_edge_feats, ncens, nlens, nbegs) #[pn,an*ifn]
        output_point_feats = tf.contrib.layers.fully_connected(
            weighted_point_feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
            activation_fn=tf.nn.relu, reuse=reuse)

        return output_point_feats


def edge_condition_diffusion_anchor(sxyzs, feats, ifn, weights_dims, ofn, anchor_num,
                                    name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        edge_weights_feats=sxyzs

        for idx,fd in enumerate(weights_dims):
            cfeats=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=fd,
                                                     scope='{}_fc_weights_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            edge_weights_feats=tf.concat([cfeats,edge_weights_feats],axis=1)

        # [en,an]
        edge_weights=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=anchor_num,
                                                       scope='{}_fc_weights_final'.format(name),
                                                       activation_fn=None, reuse=reuse)
        edge_weights=tf.clip_by_value(edge_weights,-10,10)
        edge_weights=tf.exp(edge_weights)
        edge_weights+=1e-5 # prevent divide by 0

        # [pn,an]
        point_weights_sum=neighbor_ops.neighbor_sum_feat_gather(edge_weights,ncens, nlens, nbegs)
        point_weights_sum=tf.expand_dims(point_weights_sum,axis=2)

        edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)

        # weight edge feats
        weighted_edge_feats=tf.expand_dims(edge_weights,axis=2)*tf.expand_dims(edge_feats,axis=1) # [e,an,ifn]
        weighted_edge_feats=tf.reshape(weighted_edge_feats,[-1,anchor_num*ifn])

        # sum to points
        weighted_point_feats=neighbor_ops.neighbor_sum_feat_gather(weighted_edge_feats,ncens, nlens, nbegs) #[pn,an*ifn]

        # rescale
        weighted_point_feats=tf.reshape(weighted_point_feats,[-1,anchor_num,ifn])
        weighted_point_feats/=point_weights_sum
        weighted_point_feats=tf.reshape(weighted_point_feats,[-1,anchor_num*ifn])

        output_point_feats = tf.contrib.layers.fully_connected(
            weighted_point_feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
            activation_fn=tf.nn.relu, reuse=reuse)

        return output_point_feats


# remove normalize
def edge_condition_diffusion_anchor_v2(sxyzs, feats, weights_dims, ofn, anchor_num, embed_dim,
                                       name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        # [pn,an*ed]
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=anchor_num*embed_dim,
                                                scope='{}_fc_embed'.format(name),
                                                activation_fn=None, reuse=reuse)
        edge_weights_feats=sxyzs

        for idx,fd in enumerate(weights_dims):
            cfeats=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=fd,
                                                     scope='{}_fc_weights_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            edge_weights_feats=tf.concat([cfeats,edge_weights_feats],axis=1)

        # [en,an]
        edge_weights=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=anchor_num,
                                                       scope='{}_fc_weights_final'.format(name),
                                                       activation_fn=tf.nn.sigmoid, reuse=reuse)

        # [en,an*ed]
        edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)
        edge_feats = tf.reshape(edge_feats,[-1,anchor_num,embed_dim]) # [en,an,ed]

        # weight edge feats
        weighted_edge_feats=tf.expand_dims(edge_weights,axis=2)*edge_feats           # [en,an,ed]
        weighted_edge_feats=tf.reshape(weighted_edge_feats,[-1,anchor_num*embed_dim])# [en,an*ed]

        # sum to points
        weighted_point_feats=neighbor_ops.neighbor_sum_feat_gather(weighted_edge_feats,ncens,nlens,nbegs) #[pn,an*ed]

        # normalize point number
        weighted_point_feats/=tf.expand_dims(tf.cast(nlens,tf.float32),axis=1)

        output_point_feats = tf.contrib.layers.fully_connected(
            weighted_point_feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
            activation_fn=tf.nn.relu, reuse=reuse)

        return output_point_feats


def edge_condition_diffusion_anchor_v3(sxyzs, feats, weights_dims, ofn, anchor_num, embed_dim,
                                       name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        # [pn,an*ed]
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=anchor_num*embed_dim,
                                                scope='{}_fc_embed'.format(name),
                                                activation_fn=None, reuse=reuse)
        edge_weights_feats=sxyzs

        for idx,fd in enumerate(weights_dims):
            cfeats=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=fd,
                                                     scope='{}_fc_weights_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            edge_weights_feats=tf.concat([cfeats,edge_weights_feats],axis=1)

        # [en,an]
        edge_weights=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=anchor_num,
                                                       scope='{}_fc_weights_final'.format(name),
                                                       activation_fn=None, reuse=reuse)

        norm=tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(edge_weights),axis=1)+1e-5),axis=1)
        edge_weights/=(norm+1e-5)

        # [en,an*ed]
        edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)
        edge_feats = tf.reshape(edge_feats,[-1,anchor_num,embed_dim]) # [en,an,ed]

        # weight edge feats
        weighted_edge_feats=tf.expand_dims(edge_weights,axis=2)*edge_feats           # [en,an,ed]
        weighted_edge_feats=tf.reshape(weighted_edge_feats,[-1,anchor_num*embed_dim])# [en,an*ed]

        # sum to points
        weighted_point_feats=neighbor_ops.neighbor_sum_feat_gather(weighted_edge_feats,ncens,nlens,nbegs) #[pn,an*ed]

        # normalize point number
        weighted_point_feats/=tf.expand_dims(tf.cast(nlens,tf.float32),axis=1)

        output_point_feats = tf.contrib.layers.fully_connected(
            weighted_point_feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
            activation_fn=None, reuse=reuse)

        return output_point_feats


def edge_condition_diffusion_anchor_v4(sxyzs, feats, ifn, weights_dims, ofn, anchor_num, name,
                                       nidxs, nlens, nbegs, ncens, l2_norm=False, reuse=None,
                                       final_activation=None, weights_activation=None, use_concat=False):
    with tf.name_scope(name):
        # [pn]
        if use_concat:
            edge_weights_feats=graph_concat_scatter(feats,nidxs,nlens,nbegs,ncens)
        else:
            edge_weights_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=True)
        edge_weights_feats = tf.concat([sxyzs,edge_weights_feats],axis=1)

        for idx,fd in enumerate(weights_dims):
            cfeats=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=fd,
                                                     scope='{}_fc_weights_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            edge_weights_feats=tf.concat([cfeats,edge_weights_feats],axis=1)

        # [en,an]
        edge_weights=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=anchor_num,
                                                       scope='{}_fc_weights_final'.format(name),
                                                       activation_fn=weights_activation, reuse=reuse)
        if l2_norm:
            norm=tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(edge_weights),axis=1)+1e-5),axis=1)
            edge_weights/=(norm+1e-5)
            with tf.variable_scope(name):
                weights_transformer = variable_on_cpu('edge_weights_trans', [1, anchor_num], tf.ones_initializer)
                edge_weights *= weights_transformer

        # [en,ifn]
        edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)

        # weight edge feats
        weighted_edge_feats=tf.expand_dims(edge_weights,axis=2)*tf.expand_dims(edge_feats,axis=1)  # [en,an,ed]
        weighted_edge_feats=tf.reshape(weighted_edge_feats,[-1,anchor_num*ifn])                    # [en,an*ed]

        # sum to points
        weighted_point_feats=neighbor_ops.neighbor_sum_feat_gather(weighted_edge_feats,ncens,nlens,nbegs) #[pn,an*ed]

        # normalize point number
        weighted_point_feats/=tf.expand_dims(tf.cast(nlens,tf.float32),axis=1)

        output_point_feats = tf.contrib.layers.fully_connected(
            weighted_point_feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
            activation_fn=final_activation, reuse=reuse)

        return output_point_feats
