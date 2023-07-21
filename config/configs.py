General_Config = {
    'general': {
        'batch_size': 2000,
        'data': -1,
        'epochs': 1,
        'validation_split': 0.1,
        'net_optim_lr': 1e-3,
        'criteo_embedding_size': 20,
        'avazu_embedding_size': 40,
        'huawei_embedding_size': 15,
    },
}
CELS_Config = {
    'CELS': {
        'c': 0.5,
        'mu': 0.8,
        'gRDA_optim_lr': 1e-3,
        'net_optim_lr': 1e-3,
        'interaction_fc_output_dim': 1,
        'validation_split': 0.1,
        'mutation_threshold': 0.2,
        'mutation_probability': 0.5,
        'mutation_step_size': 10,
        'adaptation_hyperparameter': 0.99,
        'adaptation_step_size': 10,
        'population_size': 4
    },
    'ModelFunctioning': {
        'interaction_fc_output_dim': 15,
        'dnn_hidden_units': [400, 400],
    }
}