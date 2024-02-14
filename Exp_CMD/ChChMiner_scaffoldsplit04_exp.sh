#!/bin/bash

echo 'ChChMiner pretrained model on scaffold split data (0-4) with GCN on ContextPred, EdgePred and MaskingNodePred; ' 
##################################################################
echo 'ChChMiner pretrained model on scaffold split data (0)' 
##################################################################
# Graph contextpred
echo '##### contextpred #####'
echo 'contextpred GCN' 
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_0.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold0/ChChMiner/gcn_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph edgepred
echo '##### EdgePred #####'
echo 'edgepred GCN'  
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_0.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold0/ChChMiner/gcn_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph Masking
echo '##### Masking #####'
echo 'Masking Node GCN'
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_0.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold0/ChChMiner/gcn_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

##################################################################
echo 'ChChMiner pretrained model on scaffold split data (1)' 
##################################################################
# Graph contextpred
echo '##### contextpred #####'
#echo 'contextpred GIN' 
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/gin_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'contextpred GCN' 
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/gcn_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'contextpred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/graphsage_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph edgepred
echo '##### EdgePred #####'
#echo 'edgepred GIN'  
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/gin_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'edgepred GCN'  
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/gcn_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'edgepred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/graphsage_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph InfoMax
#echo '##### InfoMax #####'
#echo 'InfoMax GIN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/gin_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax CCN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/gcn_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/graphsage_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# ChChMiner Graph Masking
echo '##### Masking #####'
#echo 'Masking Node GIN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/gin_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'Masking Node GCN'
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/gcn_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'Masking Node GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_1.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold1/ChChMiner/graphsage_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

##################################################################
echo 'ChChMiner pretrained model on scaffold split data (2)' 
##################################################################
# Graph contextpred
echo '##### contextpred #####'
#echo 'contextpred GIN' 
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/gin_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'contextpred GCN' 
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/gcn_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'contextpred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/graphsage_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph edgepred
echo '##### EdgePred #####'
#echo 'edgepred GIN'  
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/gin_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'edgepred GCN'  
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/gcn_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'edgepred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/graphsage_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph InfoMax
#echo '##### InfoMax #####'
#echo 'InfoMax GIN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/gin_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax CCN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/gcn_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/graphsage_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# ChChMiner Graph Masking
echo '##### Masking #####'
#echo 'Masking Node GIN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/gin_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'Masking Node GCN'
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/gcn_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'Masking Node GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_2.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold2/ChChMiner/graphsage_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

##################################################################
echo 'ChChMiner pretrained model on scaffold split data (3)' 
##################################################################
# Graph contextpred
echo '##### contextpred #####'
#echo 'contextpred GIN' 
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/gin_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'contextpred GCN' 
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/gcn_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'contextpred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/graphsage_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph edgepred
echo '##### EdgePred #####'
#echo 'edgepred GIN'  
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/gin_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'edgepred GCN'  
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/gcn_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'edgepred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/graphsage_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph InfoMax
echo '##### InfoMax #####'
#echo 'InfoMax GIN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/gin_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax CCN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/gcn_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/graphsage_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# ChChMiner Graph Masking
echo '##### Masking #####'
#echo 'Masking Node GIN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/gin_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'Masking Node GCN'
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/gcn_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'Masking Node GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_3.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold3/ChChMiner/graphsage_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

##################################################################
echo 'ChChMiner pretrained model on scaffold split data (4)' 
##################################################################
# Graph contextpred
echo '##### contextpred #####'
#echo 'contextpred GIN' 
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/gin_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'contextpred GCN' 
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/gcn_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'contextpred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_contextpred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/graphsage_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph edgepred
echo '##### EdgePred #####'
#echo 'edgepred GIN'  
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/gin_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'edgepred GCN'  
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/gcn_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'edgepred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_edgepred_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/graphsage_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph InfoMax
echo '##### InfoMax #####'
#echo 'InfoMax GIN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/gin_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax CCN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/gcn_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_infomax_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/graphsage_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# ChChMiner Graph Masking
echo '##### Masking #####'
#echo 'Masking Node GIN'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/gin_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'Masking Node GCN'
python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/gcn_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'Masking Node GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ChChMiner_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ChChMiner_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ChChMiner_test_scaffold_4.csv --vocab_path datachem/drug_list_miner.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_masking_seed_0.pth --use_gpu --gpu_id 0 --model_result_path nocl_model_res_exp3/scaffold4/ChChMiner/graphsage_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2
