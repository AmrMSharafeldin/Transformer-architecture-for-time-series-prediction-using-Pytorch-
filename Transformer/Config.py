Parameters = {
"val_size":  0.2,
"n_seq_in":10, 
"n_features" : 1 ,
"n_seq_out": 5,
"features" :["close"],
"head_size" :128,
"num_heads" :4,
"ff_dim":256,
"dropout" : 0.1,
"num_encoder_block" : 4,
"mlp_units":[256],

"batch_size":32,
"epochs" :25,
"lr":0.001,
"model_save_path": "Transformer/trained_model/transformer.pth",
"visualization_save_path": "Transformer/visualization/test_plot.jpg",
}
