#! /bin/bash

angle=("1.57" "1.47" "1.37" "1.27" "1.17" "1.67" "1.77" "1.87" "1.97" "2.07")

Nactor=1
final_time=30
init_robot_pose_x=-1
init_robot_pose_y=-1
init_robot_pose_theta=1.57
# seed=(60 61 62 63 64 65 66 67 68 69 70 71 72 73 74)
seed=(1)

dt=0.1
pred_step_size_multiplier=1
is_visualization=1 
buffer_size=25 
Ncritic=25 
gamma=0.9
critic_struct=quad-mix
circle_x_pos=-0.6
circle_y_pos=-0.5
circle_sigma=0.1
critic_period_multiplier=1

####################################################################################
ctrl_mode=CALF #N_CTRL, MPC, CALF, SARSA-m

Nruns=15

for ((i=0; i<${#seed[@]}; i++)); do

    echo -e "\033[32m ======================= TEST Number: $((k=$k+1)) ======================= \033[0m"

    python3 PRESET_3wrobot_NI.py --dt $dt --Nactor $Nactor --pred_step_size_multiplier $pred_step_size_multiplier \
    --ctrl_mode $ctrl_mode --Nruns $Nruns --init_robot_pose_x $init_robot_pose_x --init_robot_pose_y $init_robot_pose_y \
    --init_robot_pose_theta $init_robot_pose_theta --t1 $final_time --circle_x $circle_x_pos --circle_y $circle_y_pos --circle_sigma $circle_sigma \
    --is_visualization $is_visualization --buffer_size $buffer_size --Ncritic $Ncritic --gamma $gamma --critic_struct $critic_struct --seed ${seed[i]} \
    --critic_period_multiplier $critic_period_multiplier
    
done

####################################################################################
ctrl_mode=SARSA-m #N_CTRL, MPC, CALF, SARSA-m

Nruns=15

gamma=0.99

for ((i=0; i<${#seed[@]}; i++)); do

    echo -e "\033[32m ======================= TEST Number: $((k=$k+1)) ======================= \033[0m"

    python3 PRESET_3wrobot_NI.py --dt $dt --Nactor $Nactor --pred_step_size_multiplier $pred_step_size_multiplier \
    --ctrl_mode $ctrl_mode --Nruns $Nruns --init_robot_pose_x $init_robot_pose_x --init_robot_pose_y $init_robot_pose_y \
    --init_robot_pose_theta $init_robot_pose_theta --t1 $final_time --circle_x $circle_x_pos --circle_y $circle_y_pos --circle_sigma $circle_sigma \
    --is_visualization $is_visualization --buffer_size $buffer_size --Ncritic $Ncritic --gamma $gamma --critic_struct $critic_struct --seed ${seed[i]} \
    --critic_period_multiplier $critic_period_multiplier
    
done

####################################################################################
ctrl_mode=N_CTRL #N_CTRL, MPC, CALF, SARSA-m

Nruns=1

for ((i=0; i<${#seed[@]}; i++)); do

    echo -e "\033[32m ======================= TEST Number: $((k=$k+1)) ======================= \033[0m"

    python3 PRESET_3wrobot_NI.py --dt $dt --Nactor $Nactor --pred_step_size_multiplier $pred_step_size_multiplier \
    --ctrl_mode $ctrl_mode --Nruns $Nruns --init_robot_pose_x $init_robot_pose_x --init_robot_pose_y $init_robot_pose_y \
    --init_robot_pose_theta $init_robot_pose_theta --t1 $final_time --circle_x $circle_x_pos --circle_y $circle_y_pos --circle_sigma $circle_sigma \
    --is_visualization $is_visualization --buffer_size $buffer_size --Ncritic $Ncritic --gamma $gamma --critic_struct $critic_struct --seed ${seed[i]} \
    --critic_period_multiplier $critic_period_multiplier
    
done

####################################################################################
ctrl_mode=MPC #N_CTRL, MPC, CALF, SARSA-m
# Nactor=(10 15 20 25 30 35 40 45 50 55 60)
Nactor=(25)
Nruns=1

for ((i=0; i<${#Nactor[@]}; i++)); do

    echo -e "\033[32m ======================= TEST Number: $((k=$k+1)) ======================= \033[0m"

    python3 PRESET_3wrobot_NI.py --dt $dt --Nactor ${Nactor[i]} --pred_step_size_multiplier $pred_step_size_multiplier \
    --ctrl_mode $ctrl_mode --Nruns $Nruns --init_robot_pose_x $init_robot_pose_x --init_robot_pose_y $init_robot_pose_y \
    --init_robot_pose_theta $init_robot_pose_theta --t1 $final_time --circle_x $circle_x_pos --circle_y $circle_y_pos --circle_sigma $circle_sigma \
    --is_visualization $is_visualization --buffer_size $buffer_size --Ncritic $Ncritic --gamma $gamma --critic_struct $critic_struct --seed $seed \
    --critic_period_multiplier $critic_period_multiplier
    
done