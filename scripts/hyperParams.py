import datajoint as dj
import os
from itertools import product
import hashlib
import subprocess

#--------------------------------------------------------------------------------#

def list_hash(values):
    """
    Returns MD5 digest hash values for a list of values
    """
    hashed = hashlib.md5()
    for v in values:
        hashed.update(str(v).encode())
    return hashed.hexdigest()

#--------------------------------------------------------------------------------#

# Definition of the schema
schema = dj.schema('ven_pl', locals())
# NOTE: passing in locals() allows schema object to have access to all tables that you define in the local
#       name space (e.g. interative session). This allows tables to refer to each other simply by their name.

#--------------------------------------------------------------------------------#

store_dir = "/src/store"

# Which parameter-combinations should be explored?
seedL = [11]#, 11]#, 10]
iterL = [5000]#, 2000]
lrL = [0.0001]
fc_unitsL = [2000]#[200, 400]#, 2000]
fc_layersL = [3]
max_samplesL = [500]
depthL = [5]
preL = ['yes']

# Grid-searches EWC/SI/XdG
lamdaL = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
          100000000000., 1000000000000.]
onlineLamdaL = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
                100000000000., 1000000000000.]
gammaL = [1.]#, 0.9, 0.8, 0.7, 0.6, 0.5]
cL = [0.001, 0.01, 0.1, 1., 10. , 100., 1000., 10000., 100000., 1000000., 10000000., 100000000.]

#--------------------------------------------------------------------------------#

# Definition of the hyper-parameter look-up table
@schema
class HyperParameters_hyperParams(dj.Lookup):
    definition = """
    # Hyper-parameter combinations
    param_id: varchar(64) # unique ID for hyper-parameter set
    ---
    seed: int             # random seed
    iters: int            # maximum number of iterations to train for per task
    lr: float             # learning rate encoder/classifier
    fc_units: int
    fc_layers: int
    ws: enum('none', 'ewc', 'path')
    lamda: float                # EWC-lambda parameter
    online: enum('yes', 'no')   # use online EWC?
    gamma: float                # online EWC-gamma parameter
    c: float                    # SI: c parameter
    max_samples: int       # max number of training samples per class
    depth: int
    pre_trained: enum('yes', 'no')
    """
    contents = map(lambda x: (list_hash(x),) + x, product(
        seedL, # seed
        iterL,  # iters
        lrL,  # lr
        fc_unitsL,  # fc-units
        fc_layersL,  # fc-layers
        ['none'], # ws (weighted_synapses)
        [5000.], # lamda (EWC)
        ['no'], # online
        [1.], # gamma (online-EWC)
        [0.1], # c (SI)
        max_samplesL, # max_samples
        depthL, # depth
        preL, # pre_trained
    ))

#--------------------------------------------------------------------------------#

# Definition of computed table
@schema
class Precision_hyperParams(dj.Computed):
    definition = """
    # Average precision (over all tasks) of the final model
    -> HyperParameters_hyperParams
    ---
    ave_prec: float
    """

    # Restrict the "auto-population" of this table according to the here specified conditions
    @property
    def key_source(self):
        return (HyperParameters_hyperParams() & 'lr>0.00000005')
#        return (HyperParameters_splitMNIST() & 'lamda>0.001' & 'c>0.000005')
#        return (HyperParameters() & 'lr>0.0005' & 'hid_size=400')

    # Function specifying what to do for "populating" this table
    def _make_tuples(self, key):

        # Extract hyper-parameters from key & table
        hyperParams = (HyperParameters_hyperParams() & key).fetch1()
        seed = hyperParams['seed']
        iters = hyperParams['iters']
        lr = hyperParams['lr']
        fc_units = hyperParams['fc_units']
        fc_layers = hyperParams['fc_layers']
        ws = hyperParams['ws']
        lamda = hyperParams['lamda']
        online = hyperParams['online']
        gamma = hyperParams['gamma']
        c_param = hyperParams['c']
        max_samples = hyperParams['max_samples']
        depth = hyperParams['depth']
        pre_trained = hyperParams['pre_trained']

        # Set flags
        flags = ""
        if ws=="ewc":
            flags += "--ewc "
        elif ws=="path":
            flags += "--si "
        if online=="yes":
            flags += "--online "
        if max_samples<500:
            flags += "--max-samples={} ".format(max_samples)
        if pre_trained=="yes":
            flags += "--pre-convE --convE-ltag=s100N "

        # Set "option statement" for program call
        opt_stat = (
            "{flags}--seed={seed} --iters={iters} --lr={lr} --fc-units={fcu} --lambda={lamda} --gamma={gamma} --c={c} "
            "--fc-layers={fcl} --depth={depth} --experiment=CIFAR100 --tasks=10 "
            "--model-dir={sdir}/models --data-dir={sdir}/datasets --plot-dir={sdir}/plots --results-dir={sdir}/results "
                .format(flags=flags, seed=seed, iters=iters, lr=lr, fcu=fc_units, depth=depth,
                        lamda=lamda, gamma=gamma, c=c_param, fcl=fc_layers,  sdir=store_dir)
        )

        # Get param-stamp
        args_stamp = ["python3 main_cl.py"] + ["--get-stamp"] + opt_stat.split()
        output = subprocess.run(args_stamp, stdout=subprocess.PIPE)
        param_stamp = output.stdout.decode().strip('\n')
        print(param_stamp)

        # Train the network with these hyper-parameters
        if not os.path.isfile('{}/results/prec-{}.txt'.format(store_dir, param_stamp)):
            print("\n\n  ...running...")
            args_run = ["python3 main_cl.py"] + opt_stat.split()
            output = subprocess.run(args_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # write output & errors to "result"-files
            out_file = open("{}/results/jjOutput-{}".format(store_dir, param_stamp), "w")
            out_file.write(output.stdout.decode())
            out_file.close()
            err_file = open("{}/results/jjError-{}".format(store_dir, param_stamp), "w")
            err_file.write(output.stderr.decode())
            err_file.close()
            #_ = subprocess.run(args_run)
        else:
            print("\n\n  already run!")

        # Read best precision
        fileName = '{}/results/prec-{}.txt'.format(store_dir, param_stamp)
        file = open(fileName)
        ave_precision = float(file.readline())
        file.close()

        # Populate the table
        key['ave_prec'] = ave_precision
        self.insert1(key)

        print("Done: ", key)
        print("")

#--------------------------------------------------------------------------------#

# Create an instance of the lookup-table class
hyper_params = HyperParameters_hyperParams()

# Create an instance of the computed-table class
precision = Precision_hyperParams()

#--------------------------------------------------------------------------------#

##########------> grid search EWC / SI hyper-parameter

# Add additional hyper-parameter combinations to the hyperParameter lookup-table
##---> EWC
toBeAdded = map(lambda x: (list_hash(x),) + x, product(
    seedL,  # seed
    iterL,  # iters
    lrL,  # lr
    fc_unitsL,  # fc-units
    fc_layersL,  # fc-layers
    ['ewc'],  # ws (weighted_synapses)
    lamdaL,  # lamda (EWC)
    ['no'],  # online
    [1.],  # gamma (online-EWC)
    [0.1],  # c (SI)
    max_samplesL,  # max_samples
    depthL,  # depth
    preL,  # pre_trained
))
hyper_params.insert(toBeAdded, skip_duplicates=True)

# Add additional hyper-parameter combinations to the hyperParameter lookup-table
##---> online-EWC
toBeAdded = map(lambda x: (list_hash(x),) + x, product(
    seedL,  # seed
    iterL,  # iters
    lrL,  # lr
    fc_unitsL,  # fc-units
    fc_layersL,  # fc-layers
    ['ewc'],  # ws (weighted_synapses)
    onlineLamdaL,  # lamda (EWC)
    ['yes'],  # online
    gammaL,  # gamma (online-EWC)
    [0.1],  # c (SI)
    max_samplesL,  # max_samples
    depthL,  # depth
    preL,  # pre_trained
))
hyper_params.insert(toBeAdded, skip_duplicates=True)

# Add additional hyper-parameter combinations to the hyperParameter lookup-table
##---> SI
toBeAdded = map(lambda x: (list_hash(x),) + x, product(
    seedL,  # seed
    iterL,  # iters
    lrL,  # lr
    fc_unitsL,  # fc-units
    fc_layersL,  # fc-layers
    ['path'],  # ws (weighted_synapses)
    [5000.],  # lamda (EWC)
    ['no'],  # online
    [1.],  # gamma (online-EWC)
    cL,  # c (SI)
    max_samplesL,  # max_samples
    depthL,  # depth
    preL,  # pre_trained
))
hyper_params.insert(toBeAdded, skip_duplicates=True)


########## Add "main options" (if new parameter-values are added later)
toBeAdded = map(lambda x: (list_hash(x),) + x, product(
    seedL,  # seed
    iterL,  # iters
    lrL,  # lr
    fc_unitsL,  # fc-units
    fc_layersL,  # fc-layers
    ['none'],  # ws (weighted_synapses)
    [5000.],  # lamda (EWC)
    ['no'],  # online
    [1.],  # gamma (online-EWC)
    [0.1],  # c (SI)
    max_samplesL,  # max_samples
    depthL,  # depth
    preL,  # pre_trained
))
hyper_params.insert(toBeAdded, skip_duplicates=True)

#--------------------------------------------------------------------------------#

# Populate the computed table (i.e., run all the models!)
precision.populate(reserve_jobs=True)

#--------------------------------------------------------------------------------#