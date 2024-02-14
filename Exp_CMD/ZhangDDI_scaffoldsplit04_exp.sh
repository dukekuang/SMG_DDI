#!/bin/bash

echo 'ZhangDDI pretrained model on scaffold split data (0-4) with GCN on ContextPred, EdgePred and MaskingNodePred; ' 
##################################################################
echo 'ZhangDDI pretrained model on scaffold split data (0)' 
##################################################################
# Graph contextpred
echo '##### contextpred #####'
#echo 'contextpred GIN' 
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/gin_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'contextpred GCN' 
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/gcn_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'contextpred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/graphsage_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph edgepred
echo '##### EdgePred #####'
#echo 'edgepred GIN'  
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/gin_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'edgepred GCN'  
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/gcn_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'edgepred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/graphsage_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph InfoMax
#echo '##### InfoMax #####'
#echo 'InfoMax GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/gin_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax CCN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/gcn_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/graphsage_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# ZhangDDI Graph Masking
echo '##### Masking #####'
#echo 'Masking Node GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/gin_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'Masking Node GCN'
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/gcn_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'Masking Node GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/graphsage_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'Masking Node & Graph GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_0.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_0.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_0.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_masking_node_graph_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold0/ZhangDDI/gin_masking_node_graph --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

##################################################################
echo 'ZhangDDI pretrained model on scaffold split data (1)' 
##################################################################
# Graph contextpred
echo '##### contextpred #####'
#echo 'contextpred GIN' 
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/gin_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'contextpred GCN' 
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/gcn_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'contextpred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/graphsage_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph edgepred
echo '##### EdgePred #####'
#echo 'edgepred GIN'  
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/gin_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'edgepred GCN'  
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/gcn_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'edgepred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/graphsage_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph InfoMax
#echo '##### InfoMax #####'
#echo 'InfoMax GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/gin_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax CCN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/gcn_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/graphsage_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# ZhangDDI Graph Masking
echo '##### Masking #####'

#echo 'Masking Node GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/gin_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'Masking Node GCN'
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/gcn_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'Masking Node GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_1.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_1.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_1.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold1/ZhangDDI/graphsage_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

##################################################################
echo 'ZhangDDI pretrained model on scaffold split data (2)' 
##################################################################
# Graph contextpred
echo '##### contextpred #####'
#echo 'contextpred GIN' 
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/gin_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'contextpred GCN' 
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/gcn_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'contextpred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/graphsage_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph edgepred
echo '##### EdgePred #####'
#echo 'edgepred GIN'  
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/gin_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'edgepred GCN'  
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/gcn_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'edgepred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/graphsage_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph InfoMax
#echo '##### InfoMax #####'
#echo 'InfoMax GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/gin_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax CCN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/gcn_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/graphsage_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# ZhangDDI Graph Masking
echo '##### Masking #####'
#echo 'Masking Node GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/gin_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'Masking Node GCN'
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/gcn_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'Masking Node GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_2.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_2.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_2.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold2/ZhangDDI/graphsage_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

##################################################################
echo 'ZhangDDI pretrained model on scaffold split data (3)' 
##################################################################
# Graph contextpred
echo '##### contextpred #####'
#echo 'contextpred GIN' 
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/gin_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'contextpred GCN' 
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/gcn_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'contextpred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/graphsage_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph edgepred
echo '##### EdgePred #####'
#echo 'edgepred GIN'  
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/gin_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'edgepred GCN'  
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/gcn_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'edgepred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/graphsage_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph InfoMax
#echo '##### InfoMax #####'
#echo 'InfoMax GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/gin_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax CCN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/gcn_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/graphsage_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# ZhangDDI Graph Masking
echo '##### Masking #####'
#echo 'Masking Node GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/gin_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'Masking Node GCN'
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/gcn_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'Masking Node GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_3.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_3.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_3.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold3/ZhangDDI/graphsage_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2


##################################################################
echo 'ZhangDDI pretrained model on scaffold split data (4)' 
##################################################################
# Graph contextpred
echo '##### contextpred #####'
#echo 'contextpred GIN' 
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/gin_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'contextpred GCN' 
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/gcn_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'contextpred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_contextpred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/graphsage_contextpred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph edgepred
echo '##### EdgePred #####'
#echo 'edgepred GIN'  
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/gin_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'edgepred GCN'  
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/gcn_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'edgepred GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_edgepred_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/graphsage_edgepred --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# Graph InfoMax
#echo '##### InfoMax #####'
#echo 'InfoMax GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/gin_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax CCN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/gcn_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'InfoMax GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_infomax_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/graphsage_infomax --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

# ZhangDDI Graph Masking
echo '##### Masking #####'
#echo 'Masking Node GIN'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gin --pretrained_gnn_path ./pretrain/pretrain_gin_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/gin_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

echo 'Masking Node GCN'
python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder gcn --pretrained_gnn_path ./pretrain/pretrain_gcn_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/gcn_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2

#echo 'Masking Node GraphSAGE'
#python task.py --data_path datachem/scaffoldsplit/ZhangDDI_train_scaffold_4.csv --separate_val_path datachem/scaffoldsplit/ZhangDDI_valid_scaffold_4.csv --separate_test_path datachem/scaffoldsplit/ZhangDDI_test_scaffold_4.csv --vocab_path datachem/drug_list_zhang.csv --graph_encoder graphsage --pretrained_gnn_path ./pretrain/pretrain_graphsage_masking_seed_0.pth --use_gpu --gpu_id 1 --model_result_path nocl_model_res_exp3/scaffold4/ZhangDDI/graphsage_masking --seed 42 --alpha_loss 1 --beta_loss 1 --gamma_loss 1 --theta_loss 2
