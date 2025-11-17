# please run first: export TF_USE_LEGACY_KERAS=1

import argparse
import ast
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
#%env TF_USE_LEGACY_KERAS=1
import torch  # must import before any other package imports tensorflow or bash will have segmentation fault
import pickle
import priors
import pymde
import scvi
import zlib
import scanpy

import numpy as np
import pandas as pd
import single_cell_models as sc
import tensorflow as tf

from callbacks import PerformanceMonitor
from datasets import load_sc_dataset
from scib_metrics.benchmark import Benchmarker
from utils import clustering_performance

# configure GPUs for pyTorch
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    use_gpu = int(np.argmax([torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)]))
else:
    use_gpu = False


import keras

print("tf.keras path:", tf.keras.__file__)
assert "tf_keras" in tf.keras.__file__  # should be true when legacy mode is active#

start = "MSxap"
end = "MSxappppx"
max_clusters = 5

trial = np.random.randint(1, 1000)
mode = "tuning"
dataset = "Celegan_path_"+start+"_"+end
seed = 1

results_dir = os.path.join('/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/codes/vampprior-mixture-model-test/origin/VampPrior-Mixture-Model/experiments/scRNA-seq', str(seed), dataset)
os.makedirs(results_dir, exist_ok=True)

adata = scanpy.read("/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/data/packer2019_preprocessed.h5ad")
lineage_path = [end[:len(start)+i] for i in range(0,len(end)-len(start)+1)]
data = adata[adata.obs["lineage"].isin(lineage_path)]

counts = tf.convert_to_tensor(data.X.toarray(), dtype=tf.float32)

b = data.obs["batch"].astype("category")
cats = list(b.cat.categories)          # keep the category order
codes = b.cat.codes.to_numpy()         # shape (n_cells,), -1 for NaN
K = len(cats)

# one-hot (n_cells, K)
onehot = np.eye(K, dtype=np.int8)[codes]
batch_id = tf.convert_to_tensor(onehot, dtype=tf.float32)

model_config = {'n_layers': 2,'hidden_dim': 64,'latent_dim': 1,'likelihood': 'zinb',"dropout_rate":0.2, "learning_rate":1e-3}
log_counts_batch = np.ma.log(tf.einsum('ij,ik->ik', tf.cast(counts, tf.float32), batch_id))
library_log_mean = np.mean(log_counts_batch, axis=0)
library_log_var = np.var(log_counts_batch, axis=0)

batch_sizes = [128]
prior_learning_ratios = [1.0]
test_cases = [
    dict(model='scVI',
         prior='VampPriorMixture',
         prior_kwargs=dict(inference='MAP-DP', prior_learning_ratio=plr, use_labels=False))
    for plr in prior_learning_ratios]

batch_size = 128
test_case = test_cases[0]

print('*** {:s} | Trial {:d} | BS {:d}/{:d} ***'.format(dataset, trial, 0 + 1, len(batch_sizes)))
trial_seed = int(zlib.crc32(str(trial * (seed or 1)).encode())) % (2 ** 32 - 1)
model_name, prior, config = 'scVI-official', 'StandardNormal', dict(batch_size=batch_size)
print('--- {:s} ---'.format(model_name))


patience = 100
max_epochs = 10000
train_ratio = 0.9

torch.manual_seed(trial_seed)
data = data.copy()
scvi.model.SCVI.setup_anndata(data, layer="counts", batch_key="batch")
model = scvi.model.SCVI(
    adata=data,
    n_hidden=model_config['hidden_dim'],
    n_latent=model_config['latent_dim'],
    n_layers=model_config['n_layers'],
    gene_likelihood=model_config['likelihood'],
    deeply_inject_covariates=False,
    log_variational=True)

model.train(
    max_epochs=max_epochs,
    train_size=train_ratio,
    batch_size=config['batch_size'],
    check_val_every_n_epoch=1,
    early_stopping=True,
    early_stopping_patience=patience,
    devices = [1],
    plan_kwargs=dict(
        #lr=model_config['learning_rate'],
        weight_decay=0.0,
        eps=1e-7,
        lr_patience=max_epochs,
        n_steps_kl_warmup=0,
        n_epochs_kl_warmup=0
    )
)

save_path = results_dir+"/trial_"+str(trial)
os.makedirs(save_path, exist_ok=True)

i_train = model.train_indices.copy()
np.save(os.path.join(save_path, 'train_indices.npy'), i_train)
i_valid = model.validation_indices.copy()
np.save(os.path.join(save_path, 'valid_indices.npy'), i_valid)
#pymde.seed(trial_seed)
embeddings = model.get_latent_representation()
np.save(os.path.join(save_path, 'embeddings.npy'), embeddings)
torch.cuda.empty_cache()



data.obsm["scVI"] = embeddings

train_data = dict(x=tf.gather(counts, i_train), s=tf.gather(batch_id, i_train))
valid_data = dict(x=tf.gather(counts, i_valid), s=tf.gather(batch_id, i_valid))

test_case = test_cases[0]
config = {**dict(batch_size=batch_size), **test_case['prior_kwargs']}
print('--- {:s} | {:s} : {:s} ---'.format(test_case['model'], test_case['prior'], str(config)))


tf.keras.utils.set_random_seed(trial_seed)
tf.config.experimental.enable_op_determinism()

# select prior
u = sc.vamp_prior_pseudo_inputs(
    count_matrix=counts,
    one_hot_batch_id=batch_id,
    num_clusters=max_clusters,
    cell_labels=labels if config.get('use_labels') else None)
latent_prior = priors.select_prior(test_case['prior'], **config, **dict(
    latent_dim=model_config['latent_dim'],
    num_clusters=max_clusters,
    u=u,
    learning_rate=model_config['learning_rate'] * (config.get('prior_learning_ratio') or 0),
))

model = sc.scVI(
    n_genes=counts.shape[1],
    n_batches=batch_id.shape[1],
    prior=latent_prior,
    use_observed_library_size=True,
    library_log_loc=library_log_mean,
    library_log_scale=library_log_var ** 0.5,
    **model_config
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate']))

hist = model.fit(
    x=train_data,
    validation_data=valid_data,
    batch_size=config['batch_size'],
    epochs=max_epochs,
    verbose=False,
    callbacks=[PerformanceMonitor(patience=patience)]
)

model.save_weights(os.path.join(save_path, 'vmmscvi_best_checkpoint'))
with open(os.path.join(save_path, 'vmmscvi_history.pkl'), 'wb') as f:
    pickle.dump(hist.history, f)

tf.keras.utils.set_random_seed(trial_seed)
vmmscvi_embeddings = model.predict(dict(x=counts, s=batch_id), batch_size=config['batch_size'])
np.save(os.path.join(save_path, 'vmmscvi_embeddings.npy'), vmmscvi_embeddings)
cluster_probs = model.cluster_probabilities(vmmscvi_embeddings)

if cluster_probs.shape[1] > 1:
    np.save(os.path.join(save_path, 'vmmscvi_cluster_probs.npy'), cluster_probs.numpy())

data.obsm["vmmscvi"] = vmmscvi_embeddings
data.write_h5ad(os.path.join(save_path, 'trained.h5ad'))